# Environmental Awareness System for Visually Impaired Users

## Overview

This Python application processes video or webcam input and provides spoken scene understanding for visually impaired users using Google Gemini Vision.

## Files

- `main.py` - entrypoint and runtime loop
- `video_processor.py` - frame extraction and sampling
- `model_inference.py` - Gemini vision inference
- `scene_interpreter.py` - message extraction and noise filtering
- `speech_output.py` - non-overlapping TTS queue
- `requirements.txt` - Python dependencies

## Setup Instructions

1. Open a terminal in the project folder:
   ```powershell
   cd "scene"
   ```

2. Create a Python virtual environment:
   ```powershell
   python -m venv venv
   ```

3. Activate the environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

5. Add your Gemini API keys:
   - Open `config.py` and set:
   ```python
   GEMINI_API_KEY = ""  # optional single fallback
   GEMINI_API_KEYS = [
       "key_1",
       "key_2",
       "key_3",
   ]
   ```
   - Or set `GOOGLE_API_KEYS` (comma-separated) / `GOOGLE_API_KEY` in the shell.

## Running the system

### Receive and run in one command

```powershell
python main.py --receive-first
```

This waits for an incoming video, saves it to `recordings/`, and then processes it immediately.

### Run directly

```powershell
python main.py
```

If recordings exist, `main.py` processes the newest recording automatically. Otherwise it uses the webcam.

### Gemini Vision (explicit)

```powershell
$env:GOOGLE_API_KEY = "your_api_key"
python main.py --source webcam --mode gemini --interval 1.0
```

### Use a video file

```powershell
python main.py --source "C:\path\to\video.mp4" --mode gemini --interval 1.0
```

## Notes

- `--display` opens a preview window.
- Press `q` to close the display window.
- The system samples frames at the configured interval to keep latency low.
- Speech output is queued to prevent overlapping audio.

## Environment Variables

- `MODE` - default model (gemini)
- `GOOGLE_API_KEYS` - comma-separated Gemini API keys
- `GOOGLE_API_KEY` - Gemini API key
