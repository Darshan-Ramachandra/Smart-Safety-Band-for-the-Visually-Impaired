# Smart Safety Band for the Visually Impaired

This repository contains the laptop-side prototypes for a smart assistive safety band:

- `scene/` - scene understanding with Gemini vision and speech output
- `Nav guidance/` - local navigation guidance with object detection, depth, and segmentation
- `Hand guidance/` - object and hand guidance assistant
- `Liquid Assist/` - liquid fill-level estimation from a Raspberry Pi capture
- `Emergency/` - GPS and button helper scripts
- `project_launcher.py` - menu launcher for the main modules

## Important: Separate Virtual Environments

Each main module has its own dependencies, so create a separate virtual environment inside each folder. The launcher expects the environment folder to be named `venv`.

Use PowerShell from the repository root.

### 1. Scene

```powershell
cd "scene"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..
```

### 2. Nav Guidance

```powershell
cd "Nav guidance"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_offline_models.py
deactivate
cd ..
```

`Nav guidance` uses local model files under `Nav guidance/weights/`. If the weights are not present, run `python download_offline_models.py` while the Nav virtual environment is active.

### 3. Hand Guidance

```powershell
cd "Hand guidance"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..
```

`Hand guidance` expects the YOLO and MediaPipe model files in the `Hand guidance/` folder. Large model files are not committed to Git by default.

### 4. Liquid Assist

```powershell
cd "Liquid Assist"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..
```

## Configuration

Before running, update local IP addresses in:

- `Hand guidance/config.json`
- `Nav guidance/config.yaml`
- `Liquid Assist/config.json`

Add Gemini API keys locally where needed, or use environment variables when supported. Do not commit real API keys.

## Run

After all four virtual environments are created, run the launcher from the repository root:

```powershell
python project_launcher.py
```

You can also run modules directly:

```powershell
cd "scene"
.\venv\Scripts\Activate.ps1
python main.py --receive-first
```

```powershell
cd "Nav guidance"
.\venv\Scripts\Activate.ps1
python run_assistive_system.py --config config.yaml
```

```powershell
cd "Hand guidance"
.\venv\Scripts\Activate.ps1
python vision_assistant.py
```

```powershell
cd "Liquid Assist"
.\venv\Scripts\Activate.ps1
python main.py
```

## Notes

- Virtual environments, cache folders, recordings, received captures, and large model artifacts are ignored by Git.
- If you need to version large model files, use Git LFS instead of normal Git.
- This is an assistive prototype, not a certified mobility aid. Test only in controlled conditions with physical safety backups.
