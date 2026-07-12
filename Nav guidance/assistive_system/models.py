import os
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, SegformerForSemanticSegmentation

# Keep Ultralytics settings/cache in workspace-friendly location to avoid permission issues.
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.abspath(".ultralytics"))
from ultralytics import YOLO


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class YoloDetector:
    def __init__(self, weights_path: str, device: str):
        self.model = YOLO(weights_path)
        self.device = 0 if "cuda" in device and torch.cuda.is_available() else "cpu"
        self.names = self.model.names

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            source=frame_bgr,
            device=self.device,
            verbose=False,
            conf=0.25,
            imgsz=640,
        )[0]
        detections: List[Detection] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(int)
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        conf = result.boxes.conf.detach().cpu().numpy().astype(float)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            class_name = str(self.names[cls[i]])
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(conf[i]),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections


class DepthEstimator:
    def __init__(self, model_path: str, device: str, local_files_only: bool = True):
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=local_files_only)
        self.device = torch.device("cuda" if "cuda" in device and torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        depth = pred[0].detach().cpu().numpy().astype(np.float32)
        return depth


class FreeSpaceSegmenter:
    def __init__(self, model_path: str, device: str, local_files_only: bool = True):
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        self.device = torch.device("cuda" if "cuda" in device and torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.id2label: Dict[int, str] = self.model.config.id2label

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=frame_bgr.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        seg = upsampled.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return seg
