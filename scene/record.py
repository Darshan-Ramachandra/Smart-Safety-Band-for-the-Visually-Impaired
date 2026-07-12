import socket
from datetime import datetime
from pathlib import Path


PORT = 5001
RECORDINGS_DIR = Path("recordings")
OUTPUT_FILE = "received_video.mp4"


def ensure_recordings_dir():
    RECORDINGS_DIR.mkdir(exist_ok=True)
    return RECORDINGS_DIR


def build_output_path():
    recordings_dir = ensure_recordings_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return recordings_dir / f"received_video_{timestamp}.mp4"


def get_latest_recording():
    recordings_dir = ensure_recordings_dir()
    candidates = sorted(
        recordings_dir.glob("received_video_*.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    legacy_file = Path(OUTPUT_FILE)
    if legacy_file.exists():
        return str(legacy_file)
    return None


def receive_video(port=PORT, output_file=None):
    output_path = Path(output_file) if output_file else build_output_path()
    server = socket.socket()
    server.bind(("0.0.0.0", port))
    server.listen(1)

    print(f"Waiting for connection on port {port}...")

    conn, addr = server.accept()
    print(f"Connected from {addr}")

    try:
        with open(output_path, "wb") as file_handle:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                file_handle.write(data)
    finally:
        conn.close()
        server.close()

    print(f"Video received and saved as {output_path}")
    return str(output_path)


if __name__ == "__main__":
    receive_video()
