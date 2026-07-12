from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class CameraConfig:
    source: int | str
    width: int
    height: int
    fps: int
    raw_source: int | str | None = None


@dataclass
class RuntimeConfig:
    device: str
    use_fp16: bool
    frame_skip: int
    segmentation_every_n: int
    depth_every_n: int
    use_fast_yolo_only: bool
    visualize: bool
    max_latency_ms_warn: int


@dataclass
class ModelsConfig:
    yolo_weights: str
    depth_model_path: str
    segmentation_model_path: str
    local_files_only: bool


@dataclass
class AssistConfig:
    target_class: str
    detection_classes: List[str]
    danger_classes: List[str]
    speak_min_interval_sec: float
    navigation_interval_sec: float
    object_interval_sec: float
    danger_interval_sec: float
    obstacle_proximity_threshold: float
    collision_box_ratio: float
    reached_depth_threshold: float
    direction_deadband_px: int
    walkable_label_keywords: List[str]


@dataclass
class OptionalVLMConfig:
    enabled: bool
    ollama_url: str
    model_name: str


@dataclass
class SystemConfig:
    camera: CameraConfig
    runtime: RuntimeConfig
    models: ModelsConfig
    assist: AssistConfig
    optional_vlm: OptionalVLMConfig


def load_config(config_path: str | Path) -> SystemConfig:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    camera_raw = dict(raw["camera"])
    camera_raw.setdefault("raw_source", None)

    return SystemConfig(
        camera=CameraConfig(**camera_raw),
        runtime=RuntimeConfig(**raw["runtime"]),
        models=ModelsConfig(**raw["models"]),
        assist=AssistConfig(**raw["assist"]),
        optional_vlm=OptionalVLMConfig(**raw["optional_vlm"]),
    )
