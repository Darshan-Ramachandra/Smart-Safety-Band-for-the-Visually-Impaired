import queue
import threading
import time

import pyttsx3


class SpeechOutput:
    def __init__(self, rate=165, volume=1.0):
        self.queue = queue.Queue()
        self.last_spoken = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def say(self, text):
        text = text.strip()
        if not text:
            return
        if self._is_duplicate(text):
            return
        self.queue.put(text)

    def _is_duplicate(self, text):
        if text == self.last_spoken:
            return True
        return False

    def _run(self):
        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.3)
            except queue.Empty:
                continue
            self._speak(text)
            self.queue.task_done()

    def _speak(self, text):
        with self.lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                self.last_spoken = text
            except Exception:
                time.sleep(0.2)

    def shutdown(self):
        self.stop_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2)
        try:
            self.engine.stop()
        except Exception:
            pass
