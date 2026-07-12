# Offline Assistive Vision System (Blind Navigation + Object Interaction)

Fully local, real-time AI assistive system using a wearable/live camera feed and laptop inference only.

## Features
- Safe navigation prompts based on free-space + obstacle proximity
- Object detection and localization (`cup`, `bottle`, `cell phone`, `chair`, `person`, `dog`)
- Fine hand-guidance prompts to reach a selected target object
- Offline text-to-speech output
- Optional local VLM context via Ollama (disabled by default)

## Project Structure
- `run_assistive_system.py`: launcher
- `config.yaml`: runtime + model + behavior configuration
- `assistive_system/system.py`: real-time loop and fusion orchestration
- `assistive_system/models.py`: YOLO + Depth Anything + SegFormer wrappers
- `assistive_system/fusion.py`: navigation/object/hand guidance logic
- `assistive_system/audio.py`: offline TTS queue
- `assistive_system/vlm.py`: optional local Ollama reasoning client
- `SYSTEM_DESIGN.md`: architecture and sensor fusion explanation

## 1) Environment Setup
```powershell
cd "Nav guidance"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Local Model Preparation (No Cloud at Runtime)
Place weights/models in:
- `weights/yolo11n.pt`
- `weights/depth-anything-small-hf/` (Hugging Face model folder)
- `weights/segformer-b0-ade20k/` (Hugging Face model folder)

`config.yaml` is set to:
- `local_files_only: true` (strict local loading)

If model folders are not present, loading will fail by design.

## 3) Run
```powershell
cd "Nav guidance"
.\venv\Scripts\Activate.ps1
python run_assistive_system.py --config config.yaml
```

For a Raspberry Pi UDP stream, `config.yaml` can use:
```yaml
camera:
  source: "udp://@:5000"
  width: 640
  height: 480
  fps: 30
```

You can also override from the command line:
```powershell
python run_assistive_system.py --config config.yaml --camera-source "udp://@:5000"
```

Example Raspberry Pi sender:
```bash
rpicam-vid -t 0 \
  --width 640 --height 480 \
  --framerate 30 \
  --codec h264 \
  --inline \
  -o udp://192.168.114.114:5000
```

Use your laptop IP in the Pi command. On the laptop receiver side, bind to the local UDP port with `udp://@:5000`.

Controls:
- `q`: quit
- `t`: change target class during runtime (console input)

## 4) Tuning for Latency
In `config.yaml`:
- Increase `frame_skip` (example: `2`)
- Increase `depth_every_n` and `segmentation_every_n` (example: `3`)
- Lower camera resolution
- Use `device: "cuda"` with an NVIDIA GPU

## 5) Offline Constraints
- No cloud API calls in core loop
- All inference runs locally on the laptop
- Optional VLM requires local Ollama endpoint (`127.0.0.1`) only

## 6) Important Safety Note
This is an assistive prototype, not a certified mobility aid. Validate in controlled spaces and retain physical safety backups.
