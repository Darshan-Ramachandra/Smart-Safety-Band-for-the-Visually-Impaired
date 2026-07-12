from __future__ import annotations

import argparse
import json
import socket
import struct
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def capture_photo(output_path: Path) -> Path:
    try:
        from picamera2 import Picamera2  # type: ignore

        camera = Picamera2()
        config = camera.create_still_configuration()
        camera.configure(config)
        camera.start()
        time.sleep(0.8)
        camera.capture_file(str(output_path))
        camera.stop()
        print(f"Captured image using Picamera2: {output_path}")
        return output_path
    except Exception as picam_exc:
        print(f"Picamera2 capture failed ({picam_exc}), trying libcamera-still...")

    cmd = ["libcamera-still", "-n", "-o", str(output_path)]
    subprocess.run(cmd, check=True)
    print(f"Captured image using libcamera-still: {output_path}")
    return output_path


def send_image_udp(image_path: Path, config: dict) -> None:
    net = config["network"]
    laptop_ip = net["laptop_ip"]
    udp_port = int(net["udp_port"])
    chunk_size = int(net.get("chunk_size", 60000))
    ack_timeout_sec = float(net.get("ack_timeout_sec", 1.0))
    max_retries = int(net.get("max_retries", 10))

    image_bytes = image_path.read_bytes()
    total_size = len(image_bytes)
    total_chunks = (total_size + chunk_size - 1) // chunk_size

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(ack_timeout_sec)

    target = (laptop_ip, udp_port)

    try:
        init_packet = f"INIT|{image_path.name}|{total_chunks}|{total_size}".encode("utf-8")

        for _ in range(max_retries):
            sock.sendto(init_packet, target)
            try:
                reply, _ = sock.recvfrom(1024)
                if reply == b"ACK_INIT":
                    break
            except socket.timeout:
                continue
        else:
            raise TimeoutError("INIT not acknowledged by laptop receiver.")

        for idx in range(total_chunks):
            start = idx * chunk_size
            end = min(start + chunk_size, total_size)
            payload = image_bytes[start:end]
            packet = struct.pack("!I", idx) + payload

            for _ in range(max_retries):
                sock.sendto(packet, target)
                try:
                    reply, _ = sock.recvfrom(1024)
                    if reply.decode("utf-8", errors="ignore") == f"ACK|{idx}":
                        break
                except socket.timeout:
                    continue
            else:
                raise TimeoutError(f"Chunk {idx} not acknowledged.")

        for _ in range(max_retries):
            sock.sendto(b"FIN", target)
            try:
                reply, _ = sock.recvfrom(1024)
                if reply == b"ACK_FIN":
                    print("Image transfer complete.")
                    return
            except socket.timeout:
                continue

        raise TimeoutError("FIN not acknowledged by laptop receiver.")
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one Raspberry Pi photo and send to laptop over UDP")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to shared config JSON")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    with tempfile.TemporaryDirectory() as temp_dir:
        capture_path = Path(temp_dir) / "capture.jpg"
        capture_photo(capture_path)
        send_image_udp(capture_path, config)


if __name__ == "__main__":
    main()
