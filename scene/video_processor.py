import time

import cv2


class VideoProcessor:
    def __init__(self, source="webcam", sample_interval=1.0, max_frames=None):
        self.source = 0 if source in ("webcam", "camera") else source
        self.sample_interval = max(0.25, float(sample_interval))
        self.max_frames = None if max_frames is None else int(max_frames)
        self.cap = None
        self._is_live = isinstance(self.source, int)

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

    def frame_generator(self):
        if self.cap is None:
            self.open()

        frame_count = 0
        last_time = 0.0
        if self._is_live:
            last_time = time.time() - self.sample_interval
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break
                current_time = time.time()
                if current_time - last_time >= self.sample_interval:
                    last_time = current_time
                    frame_count += 1
                    yield frame, current_time
                    if self.max_frames and frame_count >= self.max_frames:
                        break
                # allow camera warm-up and responsive shutdown
                if cv2.waitKey(1) == 27:
                    break
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            step = max(int(fps * self.sample_interval), 1)
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break
                if frame_count % step == 0:
                    timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    yield frame, timestamp
                    if self.max_frames and (frame_count // step + 1) >= self.max_frames:
                        break
                frame_count += 1

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
