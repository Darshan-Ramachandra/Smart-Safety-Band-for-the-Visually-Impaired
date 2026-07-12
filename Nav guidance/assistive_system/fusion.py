from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .models import Detection


@dataclass
class ObjectState:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center_xy: Tuple[int, int]
    proximity_score: float
    side: str


@dataclass
class DangerAssessment:
    level: str
    indicator_text: str
    alert_text: str | None


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    vmin = float(np.percentile(arr, 1))
    vmax = float(np.percentile(arr, 99))
    if vmax - vmin < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_proximity_map(depth_map: np.ndarray) -> np.ndarray:
    depth_norm = _safe_normalize(depth_map)
    h, w = depth_norm.shape
    top = depth_norm[: max(1, h // 5), w // 3 : 2 * w // 3].mean()
    bottom = depth_norm[4 * h // 5 :, w // 3 : 2 * w // 3].mean()

    # Auto-infer orientation. If bottom is larger than top, larger values are likely closer.
    if bottom > top:
        proximity = depth_norm
    else:
        proximity = 1.0 - depth_norm
    return np.clip(proximity, 0.0, 1.0).astype(np.float32)


def walkable_mask_from_segmentation(
    seg_map: np.ndarray, id2label: Dict[int, str], walkable_keywords: List[str]
) -> np.ndarray:
    walkable_classes = set()
    for class_id, label in id2label.items():
        low = label.lower()
        if any(k in low for k in walkable_keywords):
            walkable_classes.add(class_id)

    if not walkable_classes:
        return np.zeros_like(seg_map, dtype=bool)

    mask = np.isin(seg_map, list(walkable_classes))

    # If the predefined floor labels are too sparse, anchor walkability to the dominant
    # bottom-center label, which is often the ground plane in egocentric navigation views.
    h, w = seg_map.shape
    anchor = seg_map[int(h * 0.72) :, int(w * 0.35) : int(w * 0.65)]
    if anchor.size and float(mask.mean()) < 0.05:
        vals, counts = np.unique(anchor, return_counts=True)
        if len(vals):
            dominant = int(vals[counts.argmax()])
            mask = np.logical_or(mask, seg_map == dominant)
    return mask


def classify_side(x_center: int, frame_w: int) -> str:
    left_band = int(frame_w * 0.4)
    right_band = int(frame_w * 0.6)
    if x_center < left_band:
        return "left"
    if x_center > right_band:
        return "right"
    return "front"


def build_object_states(
    detections: List[Detection],
    proximity_map: np.ndarray,
    frame_shape: Tuple[int, int, int],
) -> List[ObjectState]:
    h, w = frame_shape[:2]
    states: List[ObjectState] = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = proximity_map[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        prox = float(np.median(roi))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        states.append(
            ObjectState(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=(x1, y1, x2, y2),
                center_xy=(cx, cy),
                proximity_score=prox,
                side=classify_side(cx, w),
            )
        )
    return states


def compute_navigation_instruction(
    walkable_mask: np.ndarray,
    proximity_map: np.ndarray,
    collision_box_ratio: float,
    obstacle_proximity_threshold: float,
) -> str:
    h, w = walkable_mask.shape
    region_top = int(h * 0.55)
    region_bottom = int(h * 0.97)
    lane_margin = int(w * 0.08)
    lane_width = max(1, (w - (2 * lane_margin)) // 3)

    if region_bottom <= region_top or lane_width <= 0:
        return "Hold position"

    lane_scores = {}
    for idx, name in enumerate(("left", "center", "right")):
        x1 = lane_margin + (idx * lane_width)
        x2 = w - lane_margin if idx == 2 else x1 + lane_width
        lane_walkable = walkable_mask[region_top:region_bottom, x1:x2]
        lane_proximity = proximity_map[region_top:region_bottom, x1:x2]
        if lane_walkable.size == 0:
            lane_scores[name] = 0.0
            continue

        walk_score = float(lane_walkable.mean())
        clear_score = 1.0 - float((lane_proximity > obstacle_proximity_threshold).mean())
        lane_scores[name] = (0.65 * walk_score) + (0.35 * clear_score)

    center_score = lane_scores["center"]
    left_score = lane_scores["left"]
    right_score = lane_scores["right"]

    if center_score >= 0.58:
        if center_score >= max(left_score, right_score) - 0.05:
            return "Path is clear, move forward"
    if left_score >= right_score + 0.08 and left_score >= 0.42:
        return "Obstacle ahead, step slightly left"
    if right_score >= left_score + 0.08 and right_score >= 0.42:
        return "Obstacle ahead, step slightly right"
    if center_score >= 0.45:
        return "Move forward slowly"
    return "Obstacle ahead, stop and reorient"


def pick_target(states: List[ObjectState], target_class: str) -> ObjectState | None:
    candidates = [s for s in states if s.class_name.lower() == target_class.lower()]
    if not candidates:
        return None
    # Closest + confident target first.
    return sorted(candidates, key=lambda s: (-s.proximity_score, -s.confidence))[0]


def target_localization_text(state: ObjectState | None) -> str:
    if state is None:
        return "Target object not in view"
    approx_meter = max(0.2, (1.0 - state.proximity_score) * 3.0)
    return (
        f"{state.class_name} is {state.side}, "
        f"about {approx_meter:.1f} meters away"
    )


def foremost_center_object_text(
    states: List[ObjectState],
    frame_shape: Tuple[int, int, int],
) -> str | None:
    h, w = frame_shape[:2]
    center_left = int(w * 0.33)
    center_right = int(w * 0.67)
    center_top = int(h * 0.18)
    center_bottom = int(h * 0.92)

    center_states = [
        s
        for s in states
        if center_left <= s.center_xy[0] <= center_right and center_top <= s.center_xy[1] <= center_bottom
    ]
    if not center_states:
        return None

    best = sorted(center_states, key=lambda s: (-s.proximity_score, -s.confidence))[0]
    article = "an" if best.class_name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"There is {article} {best.class_name} ahead"


def yolo_proximity_score(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> float:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    area_score = min(1.0, area / float(max(1, int(0.22 * w * h))))
    bottom_score = min(1.0, y2 / float(max(1, h)))
    return float((0.65 * area_score) + (0.35 * bottom_score))


def build_object_states_fast(
    detections: List[Detection],
    frame_shape: Tuple[int, int, int],
) -> List[ObjectState]:
    h, w = frame_shape[:2]
    states: List[ObjectState] = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        states.append(
            ObjectState(
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=(x1, y1, x2, y2),
                center_xy=(cx, cy),
                proximity_score=yolo_proximity_score((x1, y1, x2, y2), frame_shape),
                side=classify_side(cx, w),
            )
        )
    return states


def compute_yolo_navigation_instruction(states: List[ObjectState], frame_shape: Tuple[int, int, int]) -> str:
    h, w = frame_shape[:2]
    center_left = int(w * 0.32)
    center_right = int(w * 0.68)
    bottom_gate = int(h * 0.35)
    ahead_states = [
        s
        for s in states
        if center_left <= s.center_xy[0] <= center_right and s.bbox[3] >= bottom_gate
    ]
    if not ahead_states:
        return "Path is clear, move forward"

    ahead_states = sorted(ahead_states, key=lambda s: (-s.proximity_score, -s.confidence))
    nearest = ahead_states[0]
    left_load = sum(s.proximity_score for s in states if s.side == "left")
    right_load = sum(s.proximity_score for s in states if s.side == "right")

    if nearest.proximity_score >= 0.72:
        if left_load + 0.12 < right_load:
            return f"{nearest.class_name} ahead, step slightly left"
        if right_load + 0.12 < left_load:
            return f"{nearest.class_name} ahead, step slightly right"
        return f"{nearest.class_name} ahead, stop"
    if left_load + 0.08 < right_load:
        return f"{nearest.class_name} ahead, keep slightly left"
    if right_load + 0.08 < left_load:
        return f"{nearest.class_name} ahead, keep slightly right"
    return f"{nearest.class_name} ahead, move carefully"


def surroundings_summary_text(states: List[ObjectState]) -> str | None:
    if not states:
        return None
    best = sorted(states, key=lambda s: (-s.proximity_score, -s.confidence))[:3]
    parts = []
    for s in best:
        if s.side == "front":
            parts.append(f"{s.class_name} ahead")
        else:
            parts.append(f"{s.class_name} on the {s.side}")
    if not parts:
        return None
    return "Nearby: " + ", ".join(parts)


def landmark_reference_text(
    seg_map: np.ndarray | None,
    id2label: Dict[int, str] | None,
    frame_shape: Tuple[int, int, int],
) -> str | None:
    if seg_map is None or id2label is None:
        return None

    h, w = seg_map.shape
    regions = {
        "left": seg_map[int(h * 0.20) : int(h * 0.90), : int(w * 0.38)],
        "front": seg_map[int(h * 0.20) : int(h * 0.90), int(w * 0.33) : int(w * 0.67)],
        "right": seg_map[int(h * 0.20) : int(h * 0.90), int(w * 0.62) :],
    }
    targets = {
        "door": {"door"},
        "window": {"windowpane", "window"},
        "bookcase": {"bookcase"},
    }

    best_match = None
    best_score = 0.0
    for side, region in regions.items():
        if region.size == 0:
            continue
        vals, counts = np.unique(region, return_counts=True)
        total = float(region.size)
        for value, count in zip(vals.tolist(), counts.tolist()):
            label = id2label.get(int(value), "").lower()
            for name, aliases in targets.items():
                if label in aliases:
                    score = count / total
                    if score > best_score and score >= 0.08:
                        best_score = score
                        best_match = (name, side)

    if best_match is None:
        return None
    name, side = best_match
    if side == "front":
        return f"There is a {name} ahead"
    return f"There is a {name} on the {side}"


def find_clear_direction(states: List[ObjectState]) -> str:
    left_load = sum(s.proximity_score for s in states if s.side == "left")
    right_load = sum(s.proximity_score for s in states if s.side == "right")
    if left_load + 0.12 < right_load:
        return "left"
    if right_load + 0.12 < left_load:
        return "right"
    return "stop"


def assess_danger(
    states: List[ObjectState],
    frame_shape: Tuple[int, int, int],
    danger_classes: List[str],
    approach_memory: Dict[str, Dict[str, float]],
) -> DangerAssessment:
    if not states:
        return DangerAssessment(level="safe", indicator_text="SAFE", alert_text=None)

    h, w = frame_shape[:2]
    center_left = int(w * 0.34)
    center_right = int(w * 0.66)
    danger_set = {name.lower() for name in danger_classes}

    highest_level = "safe"
    best_alert = None
    best_score = -1.0

    for s in states:
        key = f"{s.class_name}:{s.side}"
        previous = approach_memory.get(key, {})
        previous_score = float(previous.get("proximity_score", s.proximity_score))
        delta = s.proximity_score - previous_score
        is_approaching = delta > 0.12
        in_center = center_left <= s.center_xy[0] <= center_right
        is_danger_class = s.class_name.lower() in danger_set
        is_animal = s.class_name.lower() in {"dog", "cat", "cow", "horse", "sheep"}
        close_blocking_object = in_center and s.proximity_score >= 0.84

        # True danger is reserved for things that are actively closing in or animals
        # behaving unpredictably near the user, not just any detected person.
        if is_danger_class and is_approaching:
            if is_animal:
                alert = f"{s.class_name} approaching from {s.side}, stay alert"
                score = 3.2 + s.proximity_score
            else:
                direction = find_clear_direction(states)
                if direction == "stop":
                    alert = f"{s.class_name} approaching ahead, stop"
                else:
                    alert = f"{s.class_name} approaching ahead, take {direction}"
                score = 3.0 + s.proximity_score
            if score > best_score:
                highest_level = "danger"
                best_score = score
                best_alert = alert
            continue

        if is_animal and s.proximity_score >= 0.74:
            alert = f"{s.class_name} nearby on the {s.side}, stay alert"
            score = 2.7 + s.proximity_score
            if score > best_score:
                highest_level = "danger"
                best_score = score
                best_alert = alert
            continue

        if close_blocking_object:
            direction = find_clear_direction(states)
            if direction == "stop":
                alert = f"{s.class_name} ahead, stop"
            else:
                alert = f"{s.class_name} ahead, take {direction}"
            score = 2.1 + s.proximity_score
            if score > best_score:
                highest_level = "caution" if highest_level != "danger" else highest_level
                best_score = score
                best_alert = alert

    if highest_level == "danger":
        return DangerAssessment(level="danger", indicator_text="DANGER", alert_text=best_alert)
    if highest_level == "caution":
        return DangerAssessment(level="caution", indicator_text="CAUTION", alert_text=best_alert)
    return DangerAssessment(level="safe", indicator_text="SAFE", alert_text=None)


def hand_guidance_text(
    target: ObjectState | None,
    frame_shape: Tuple[int, int, int],
    direction_deadband_px: int,
    reached_depth_threshold: float,
) -> str:
    if target is None:
        return "Search with your hand near center"

    h, w = frame_shape[:2]
    tx, ty = target.center_xy
    dx = tx - (w // 2)
    dy = ty - int(h * 0.62)
    cmds: List[str] = []

    if abs(dx) > direction_deadband_px:
        cmds.append("right" if dx > 0 else "left")
    if abs(dy) > direction_deadband_px:
        cmds.append("down" if dy > 0 else "up")

    if target.proximity_score >= reached_depth_threshold:
        return "Stop, object reached"
    if not cmds:
        return "Move hand forward"
    joined = " and ".join(cmds)
    return f"Slightly {joined}, then forward"
