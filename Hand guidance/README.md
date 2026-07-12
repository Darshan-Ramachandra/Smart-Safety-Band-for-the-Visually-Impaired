# Hand Guidance

Hand Guidance combines YOLO object detection and MediaPipe hand landmarks to guide a user toward nearby objects.

## Files

- `vision_assistant.py` - main assistant loop
- `instant.py` - quick helper script
- `config.json` - camera stream configuration
- `prompt.md` - assistant prompt/reference text
- `requirements.txt` - Python dependencies

## Setup

Create this module's own virtual environment:

```powershell
cd "Hand guidance"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place required local model files in this folder, such as YOLO weights and `hand_landmarker.task`. Large model files are not committed to Git by default.

## Run

```powershell
.\venv\Scripts\Activate.ps1
python vision_assistant.py
```

Edit `config.json` if the Raspberry Pi stream IP or port changes.
