import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np
import torch

from .audio import OfflineSpeaker
from .config import SystemConfig, load_config
from .fusion import (
    DangerAssessment,
    assess_danger,
    build_object_states,
    build_object_states_fast,
    compute_navigation_instruction,
    compute_yolo_navigation_instruction,
    foremost_center_object_text,
    landmark_reference_text,
    make_proximity_map,
    pick_target,
    surroundings_summary_text,
    target_localization_text,
    walkable_mask_from_segmentation,
)
from .models import DepthEstimator, FreeSpaceSegmenter, YoloDetector
from .vlm import LocalVLMReasoner


def sanitize_broken_proxy_env() -> None:
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"]
    broken_tokens = ("127.0.0.1:9", "localhost:9")
    cleared = []
    for k in proxy_keys:
        val = os.environ.get(k) or os.environ.get(k.lower())
        if val and any(t in val for t in broken_tokens):
            os.environ.pop(k, None)
            os.environ.pop(k.lower(), None)
            cleared.append(k)
    if cleared:
        print(f"[INFO] Cleared broken proxy variables: {', '.join(cleared)}")


class LatestFrameReader:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._alive = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while self._alive:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def close(self) -> None:
        self._alive = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class AssistiveVisionSystem:
    def __init__(self, config: SystemConfig):
        self.cfg = config
        self._requested_camera_source = config.camera.raw_source or config.camera.source
        self._opened_camera_source: int | str | None = None
        self.detector = YoloDetector(config.models.yolo_weights, config.runtime.device)
        self.depth_model = None
        self.segmenter = None
        self._depth_available = True
        self._seg_available = True
        try:
            self.depth_model = DepthEstimator(
                config.models.depth_model_path,
                config.runtime.device,
                config.models.local_files_only,
            )
        except Exception as e:
            print(f"[WARN] Depth model unavailable. Using fallback proximity mode. Details: {e}")
            self._depth_available = False

        try:
            self.segmenter = FreeSpaceSegmenter(
                config.models.segmentation_model_path,
                config.runtime.device,
                config.models.local_files_only,
            )
        except Exception as e:
            print(f"[WARN] Segmentation model unavailable. Using fallback walkable mode. Details: {e}")
            self._seg_available = False
        self.speaker = OfflineSpeaker(config.assist.speak_min_interval_sec)
        self.vlm = None
        if config.optional_vlm.enabled:
            self.vlm = LocalVLMReasoner(config.optional_vlm.ollama_url, config.optional_vlm.model_name)

        self.last_nav_t = 0.0
        self.last_obj_t = 0.0
        self.last_danger_t = 0.0
        self.last_vlm_t = 0.0
        self._last_spoken_nav = ""
        self._last_spoken_ahead = ""
        self._last_spoken_landmark = ""
        self._last_spoken_summary = ""
        self._last_spoken_danger = ""
        self._detection_class_set = {x.lower() for x in self.cfg.assist.detection_classes}
        self._approach_memory: Dict[str, Dict[str, float]] = {}

        self._last_depth: np.ndarray | None = None
        self._last_seg: np.ndarray | None = None

    def _fallback_depth(self, frame: np.ndarray) -> np.ndarray:
        # Near-ground heuristic: pixels lower in frame are likely closer.
        h, w = frame.shape[:2]
        grad = np.linspace(0.1, 1.0, h, dtype=np.float32).reshape(h, 1)
        return np.repeat(grad, w, axis=1)

    def _fallback_walkable(self, frame: np.ndarray, detections) -> np.ndarray:
        # Default to walkable and carve detections as non-walkable obstacles.
        h, w = frame.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = False
        return mask

    def _open_camera(self) -> cv2.VideoCapture:
        source = self.cfg.camera.raw_source
        if source is None:
            source = self._normalize_camera_source(self.cfg.camera.source)
        else:
            source = self._coerce_camera_source(source)
        self._opened_camera_source = source
        if isinstance(source, str) and source.startswith("udp://"):
            source = self._format_udp_source(source)
            self._opened_camera_source = source
            print(f"[INFO] Opening UDP camera source: {source}")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            print(f"[INFO] Opening camera source: {source}")
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.camera.fps)
        return cap

    def _normalize_camera_source(self, source: int | str) -> int | str:
        source = self._coerce_camera_source(source)
        if isinstance(source, int):
            return source

        if not source.startswith("udp://"):
            return source

        parts = urlsplit(source)
        host = parts.hostname or ""
        if not host or host in {"0.0.0.0", "127.0.0.1", "localhost"}:
            return source

        # For inbound Raspberry Pi UDP streams, bind locally to the port instead of
        # treating the Pi IP as the address to open.
        local_netloc = f"@:{parts.port}" if parts.port is not None else "@"
        return urlunsplit((parts.scheme, local_netloc, parts.path, parts.query, parts.fragment))

    def _coerce_camera_source(self, source: int | str) -> int | str:
        if isinstance(source, int):
            return source

        source = source.strip()
        if source.isdigit():
            return int(source)
        return source

    def _format_udp_source(self, source: str) -> str:
        parts = urlsplit(source)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("overrun_nonfatal", "1")
        query.setdefault("fifo_size", "50000000")
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    def _camera_open_error(self) -> RuntimeError:
        requested = self._requested_camera_source
        opened = self._opened_camera_source
        if isinstance(requested, str) and requested.startswith("udp://"):
            return RuntimeError(
                "Could not receive the Raspberry Pi UDP stream. "
                f"Requested source: {requested}. Opened as: {opened}. "
                "Make sure the Pi is actively streaming H.264 to this laptop's IP on the same port, "
                "and that Windows Firewall is not blocking UDP traffic."
            )
        return RuntimeError(f"Could not open camera source: {opened}")

    def _overlay(self, frame, states, walkable_mask, nav_text, obj_text, latency_ms, danger: DangerAssessment):
        vis = frame.copy()
        for s in states:
            x1, y1, x2, y2 = s.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (45, 220, 60), 2)
            t = f"{s.class_name} {s.confidence:.2f} prox:{s.proximity_score:.2f}"
            cv2.putText(vis, t, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 220, 60), 1)

        # Walkable overlay (blue tint).
        if walkable_mask is not None:
            tint = np.zeros_like(vis)
            tint[:, :, 0] = 160
            vis = np.where(walkable_mask[..., None], cv2.addWeighted(vis, 0.65, tint, 0.35, 0), vis)

        cv2.putText(vis, f"NAV: {nav_text}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(vis, f"OBJ: {obj_text}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
        lat_color = (40, 255, 40) if latency_ms <= self.cfg.runtime.max_latency_ms_warn else (0, 0, 255)
        cv2.putText(vis, f"Latency: {latency_ms:.1f} ms", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lat_color, 2)

        if danger.level != "safe":
            if danger.level == "danger":
                badge_color = (0, 0, 255)
            else:
                badge_color = (0, 165, 255)
            cv2.rectangle(vis, (vis.shape[1] - 160, 16), (vis.shape[1] - 20, 56), badge_color, -1)
            cv2.putText(
                vis,
                danger.indicator_text,
                (vis.shape[1] - 148, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
        return vis

    def run(self):
        cap = self._open_camera()
        if not cap.isOpened():
            raise self._camera_open_error()
        reader = LatestFrameReader(cap)

        frame_idx = 0
        nav_text = "Initializing navigation"
        obj_text = "Initializing object localization"
        danger = DangerAssessment(level="safe", indicator_text="SAFE", alert_text=None)

        try:
            first_frame_deadline = time.time() + 3.0
            while True:
                ok, frame = reader.read()
                if not ok or frame is None:
                    if time.time() > first_frame_deadline:
                        raise self._camera_open_error()
                    time.sleep(0.01)
                    continue
                start_t = time.time()
                frame_idx += 1
                if self.cfg.runtime.frame_skip > 1 and (frame_idx % self.cfg.runtime.frame_skip != 0):
                    continue

                run_depth = self._depth_available and (
                    (not self.cfg.runtime.use_fast_yolo_only)
                    and (self._last_depth is None or (frame_idx % self.cfg.runtime.depth_every_n == 0))
                )
                run_seg = self._seg_available and (
                    (not self.cfg.runtime.use_fast_yolo_only)
                    and (self._last_seg is None or (frame_idx % self.cfg.runtime.segmentation_every_n == 0))
                )

                with ThreadPoolExecutor(max_workers=2) as ex:
                    depth_future = ex.submit(self.depth_model.infer, frame) if run_depth else None
                    seg_future = ex.submit(self.segmenter.infer, frame) if run_seg else None

                    detections = self.detector.infer(frame)
                    if depth_future is not None:
                        self._last_depth = depth_future.result()
                    if seg_future is not None:
                        self._last_seg = seg_future.result()

                filtered = [d for d in detections if d.class_name.lower() in self._detection_class_set]
                if self.cfg.runtime.use_fast_yolo_only:
                    states = build_object_states_fast(filtered, frame.shape)
                    walkable_mask = self._fallback_walkable(frame, filtered)
                    target = None
                    proximity_map = None
                    if self._seg_available and self.segmenter is not None and (
                        self._last_seg is None or (frame_idx % max(1, self.cfg.runtime.segmentation_every_n) == 0)
                    ):
                        self._last_seg = self.segmenter.infer(frame)
                else:
                    if self._last_depth is None:
                        self._last_depth = self._fallback_depth(frame)
                    if self._last_seg is None:
                        self._last_seg = np.zeros(frame.shape[:2], dtype=np.uint8)

                    proximity_map = make_proximity_map(self._last_depth)
                    if self._seg_available and self.segmenter is not None:
                        walkable_mask = walkable_mask_from_segmentation(
                            self._last_seg,
                            self.segmenter.id2label,
                            self.cfg.assist.walkable_label_keywords,
                        )
                    else:
                        walkable_mask = self._fallback_walkable(frame, filtered)
                    states = build_object_states(filtered, proximity_map, frame.shape)
                    target = pick_target(states, self.cfg.assist.target_class)

                ahead_text = foremost_center_object_text(states, frame.shape)
                landmark_text = None
                if self._seg_available and self.segmenter is not None:
                    landmark_text = landmark_reference_text(self._last_seg, self.segmenter.id2label, frame.shape)
                summary_text = surroundings_summary_text(states)
                danger = assess_danger(
                    states,
                    frame.shape,
                    self.cfg.assist.danger_classes,
                    self._approach_memory,
                )

                next_memory: Dict[str, Dict[str, float]] = {}
                for s in states:
                    next_memory[f"{s.class_name}:{s.side}"] = {"proximity_score": s.proximity_score}
                self._approach_memory = next_memory

                now = time.time()
                if (
                    danger.alert_text
                    and now - self.last_danger_t >= self.cfg.assist.danger_interval_sec
                    and danger.alert_text != self._last_spoken_danger
                ):
                    self.speaker.say(danger.alert_text)
                    self._last_spoken_danger = danger.alert_text
                    self.last_danger_t = now

                if now - self.last_nav_t >= self.cfg.assist.navigation_interval_sec:
                    if self.cfg.runtime.use_fast_yolo_only:
                        nav_text = danger.alert_text or compute_yolo_navigation_instruction(states, frame.shape)
                    else:
                        nav_text = compute_navigation_instruction(
                            walkable_mask,
                            proximity_map,
                            self.cfg.assist.collision_box_ratio,
                            self.cfg.assist.obstacle_proximity_threshold,
                        )
                    if nav_text != self._last_spoken_nav:
                        self.speaker.say(nav_text)
                        self._last_spoken_nav = nav_text
                    self.last_nav_t = now

                if now - self.last_obj_t >= self.cfg.assist.object_interval_sec:
                    obj_text = ahead_text or landmark_text or summary_text or target_localization_text(target)
                    if ahead_text and ahead_text != self._last_spoken_ahead:
                        self.speaker.say(ahead_text)
                        self._last_spoken_ahead = ahead_text
                    elif landmark_text and landmark_text != self._last_spoken_landmark:
                        self.speaker.say(landmark_text)
                        self._last_spoken_landmark = landmark_text
                    elif summary_text and summary_text != self._last_spoken_summary:
                        self.speaker.say(summary_text)
                        self._last_spoken_summary = summary_text
                    elif target is not None:
                        self.speaker.say(obj_text)
                    self.last_obj_t = now

                if self.vlm is not None and now - self.last_vlm_t > 3.0:
                    prompt = (
                        f"Detected objects: {[s.class_name for s in states]}. "
                        f"Navigation: {nav_text}. Danger: {danger.level}. Scene summary: {obj_text}. "
                        "Give one short tactical suggestion."
                    )
                    tip = self.vlm.summarize(prompt)
                    if tip:
                        self.speaker.say(tip)
                    self.last_vlm_t = now

                latency_ms = (time.time() - start_t) * 1000.0
                if self.cfg.runtime.visualize:
                    vis = self._overlay(frame, states, walkable_mask, nav_text, obj_text, latency_ms, danger)
                    cv2.imshow("Offline Assistive Vision System", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("t"):
                        self.cfg.assist.target_class = input("Enter new target class: ").strip() or self.cfg.assist.target_class
        finally:
            reader.close()
            self.speaker.close()
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Offline AI assistive system for blind users.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--camera-source",
        type=str,
        default=None,
        help="Optional camera source override. Use 0 for laptop webcam, 1/2 for external cams, or a video path.",
    )
    args = parser.parse_args()

    sanitize_broken_proxy_env()
    cfg = load_config(Path(args.config))
    if args.camera_source is not None:
        cfg.camera.source = int(args.camera_source) if args.camera_source.isdigit() else args.camera_source
    if cfg.runtime.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        cfg.runtime.device = "cpu"

    system = AssistiveVisionSystem(cfg)
    system.run()


if __name__ == "__main__":
    main()
