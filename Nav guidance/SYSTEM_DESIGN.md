# Offline AI Assistive System Design (Wearable Camera + Laptop)

## 1) End-to-End Goal
Build a fully local assistive vision system for blind users that:
- Gives continuous safe navigation instructions
- Detects and localizes specific objects
- Guides hand movement for final interaction
- Runs in real time on a laptop with no cloud APIs

## 2) Core Model Pipeline (All Local)
- **Object detection**: YOLO (Ultralytics)
  - Input: RGB frame
  - Output: `(class, confidence, bbox)`
- **Monocular depth**: Depth Anything
  - Input: RGB frame
  - Output: dense depth map
- **Free-space segmentation**: SegFormer-B0
  - Input: RGB frame
  - Output: semantic class map
  - Converted to walkable/non-walkable binary mask by class keywords (`floor`, `road`, `sidewalk`, etc.)

## 3) Sensor Fusion
For each frame:
1. Run YOLO, Depth, Segmentation (Depth + Seg can run in parallel threads)
2. Convert depth map to **proximity map** (0 far, 1 near) with automatic inversion check
3. Create walkable mask from segmentation labels
4. For each detection box:
   - Compute median proximity in bbox
   - Infer object side (`left/right/front`) from bbox center
5. Compute:
   - Collision risk in center corridor
   - Best safe direction from left/right walkable scores
   - Target object closest instance

## 4) Behavior Logic
- **Navigation**:
  - If center corridor mostly walkable and near-obstacle ratio is low -> "Path is clear, move forward"
  - Else steer toward side with higher walkable score
- **Object localization**:
  - Example: "Bottle is right, about 1.1 meters away"
- **Hand guidance**:
  - Uses target bbox center vs preferred hand anchor zone
  - Generates directional corrections (`left/right/up/down/forward`)
  - Stops when proximity exceeds reach threshold

## 5) Optional Context Layer
- Optional local Ollama + Qwen2.5-VL-3B call for short high-level tactical suggestions
- Not used for core safety loop; core loop stays deterministic

## 6) Real-Time Performance Strategy
- Frame skipping (`frame_skip`)
- Run depth/segmentation every N frames (`depth_every_n`, `segmentation_every_n`)
- Keep YOLO at lower `imgsz` for speed
- Use GPU + FP16 when available
- Warn when latency exceeds threshold (`max_latency_ms_warn`)

## 7) Safety Considerations
- This is an assistive prototype and must be validated in controlled spaces first
- Keep conservative thresholds and prefer "stop/reorient" over aggressive movement
- Add hardware fallback (white cane, human supervision) in real deployments

