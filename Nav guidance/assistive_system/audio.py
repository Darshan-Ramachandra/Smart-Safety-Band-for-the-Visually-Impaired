import queue
import threading
import time

import pyttsx3


class OfflineSpeaker:
    def __init__(self, min_interval_sec: float = 1.0):
        self.engine = None
        self.enabled = True
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 180)
        except Exception as e:
            print(f"[WARN] Offline TTS unavailable. Continuing without speech. Details: {e}")
            self.enabled = False
        self.min_interval_sec = min_interval_sec
        self._last_spoken = 0.0
        self._last_text = ""
        self._current_text = ""
        self._repeat_count = 0
        self._queue: queue.Queue[str] = queue.Queue()
        self._alive = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while self._alive:
            try:
                text = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._current_text = text
            if self.enabled and self.engine is not None:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception:
                    self.enabled = False

    def _drain_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            return

    def say(self, text: str) -> None:
        if not self.enabled:
            return
        now = time.time()
        if (now - self._last_spoken) < self.min_interval_sec:
            return

        if text == self._last_text:
            if self._repeat_count >= 2:
                return
            self._repeat_count += 1
        else:
            self._repeat_count = 1

        if text == self._current_text:
            return

        # Keep only the freshest message so speech never builds a long backlog.
        self._drain_queue()
        self._last_text = text
        self._last_spoken = now
        self._queue.put(text)

    def close(self) -> None:
        self._alive = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
