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

### Local IP addresses

If you run through `project_launcher.py`, do not manually edit the IP first. The launcher reads the saved Raspberry Pi IP, asks whether it is still correct, and updates these files for you when you answer `n`:

- `Hand guidance/config.json`
- `Nav guidance/config.yaml`
- `Liquid Assist/config.json`

If you run a module directly without the launcher, update the Raspberry Pi IP manually in these exact locations:

- `Hand guidance/config.json`, line 2: `stream_url`
- `Nav guidance/config.yaml`, lines 3-4: `camera.source` and `camera.raw_source`
- `Liquid Assist/config.json`, line 3: `network.raspberry_pi_ip`

### Gemini API keys

Do not commit real API keys. Add keys only in your local working copy or set environment variables when the module supports them.

`scene/` supports environment variables, CLI input, and local config:

- Preferred temporary PowerShell option: `$env:GOOGLE_API_KEY="your-key"` or `$env:GOOGLE_API_KEYS="key-1,key-2"`
- Direct run option: `scene/main.py`, lines 53-56: `--google-api-key`
- Local config option: `scene/config.py`, lines 1-4: `GEMINI_API_KEY` or `GEMINI_API_KEYS`

`Liquid Assist/` currently reads Gemini keys from local config only:

- `Liquid Assist/config.json`, line 16: add keys inside `gemini.api_keys`, for example `["key-1", "key-2"]`

`Nav guidance/` and `Hand guidance/` do not use Gemini keys.

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
