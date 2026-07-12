# Liquid Assist

Liquid Assist captures **one photo** on Raspberry Pi, sends it to the laptop using UDP (with ACK/retry), and runs Gemini fill-level estimation.

## Files

- `main.py`: Laptop receiver + Gemini analyzer
- `pi_capture_sender.py`: Raspberry Pi one-photo capture + UDP sender
- `config.json`: Shared IP/port/API settings
- `requirements.txt`: Python dependencies for laptop analyzer

## Laptop steps

1. Create and activate this module's virtual environment:
   ```powershell
   cd "Liquid Assist"
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Run from launcher (`project_launcher.py`) and choose `4. Liquid Assist`.
   - It waits for one incoming image, saves `received_capture.jpg`, then prints result.

## Raspberry Pi steps

1. Copy `pi_capture_sender.py` + same `config.json` to Pi.
2. Ensure `network.laptop_ip` points to laptop IP.
3. Run:
   ```bash
   python3 pi_capture_sender.py
   ```

## Config

Edit `Liquid Assist/config.json`:
- `network.raspberry_pi_ip`: IP of Raspberry Pi sender
- `network.laptop_ip`: laptop bind/sender target IP
- `network.udp_port`: UDP port used for photo transfer
- `gemini.api_keys`: multiple keys for fallback
