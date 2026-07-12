from __future__ import annotations

import argparse
import json
import socket
import time
import warnings
from pathlib import Path

from PIL import Image
import pyttsx3

# Suppress deprecated Gemini SDK warnings in terminal output.
warnings.simplefilter("ignore", FutureWarning)
import google.generativeai as genai

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


class GeminiFallbackClient:
    def __init__(self, api_keys: list[str], model_name: str) -> None:
        if not api_keys:
            raise ValueError("No Gemini API keys configured.")
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0
        self._configure_current_key()

    def _configure_current_key(self) -> None:
        genai.configure(api_key=self.api_keys[self.current_key_index])
        self.model = genai.GenerativeModel(self.model_name)

    def _try_next_api_key(self) -> None:
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._configure_current_key()

    @staticmethod
    def _safe_gemini_text(response) -> str | None:
        try:
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            return text.strip()
        except Exception:
            return None
        return None

    def generate_with_fallback(self, prompt: str, image: Image.Image) -> str | None:
        for _ in range(len(self.api_keys)):
            try:
                response = self.model.generate_content([prompt, image])
                text = self._safe_gemini_text(response)
                if text:
                    return text
            except Exception:
                pass
            self._try_next_api_key()
        return None


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def recv_exact(conn: socket.socket, num_bytes: int) -> bytes:
    data = b""
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet:
            raise ConnectionError("Connection closed before all bytes were received.")
        data += packet
    return data


def receive_image_tcp(config: dict, output_path: Path) -> Path:
    net = config["network"]
    expected_pi_ip = net["raspberry_pi_ip"]
    bind_ip = net["laptop_ip"]
    tcp_port = int(net.get("tcp_port", net.get("udp_port", 5055)))
    receive_timeout_sec = float(net["receive_timeout_sec"])

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((bind_ip, tcp_port))
    server.listen(1)
    server.settimeout(receive_timeout_sec)
    print(f"Waiting for TCP image on {bind_ip}:{tcp_port} from Raspberry Pi {expected_pi_ip}...")

    try:
        while True:
            conn, addr = server.accept()
            sender_ip, _ = addr
            if sender_ip != expected_pi_ip:
                conn.close()
                continue

            with conn:
                name_len = int.from_bytes(recv_exact(conn, 4), "big")
                file_name = recv_exact(conn, name_len).decode("utf-8")
                file_size = int.from_bytes(recv_exact(conn, 8), "big")
                print(f"Receiving {file_name} ({file_size} bytes)...")

                bytes_received = 0
                with open(output_path, "wb") as file_handle:
                    while bytes_received < file_size:
                        chunk = conn.recv(min(8192, file_size - bytes_received))
                        if not chunk:
                            raise ConnectionError("Connection closed during file transfer.")
                        file_handle.write(chunk)
                        bytes_received += len(chunk)

                if bytes_received != file_size:
                    raise RuntimeError("Transferred image size mismatch.")

            print(f"Image saved to {output_path}")
            return output_path
    finally:
        server.close()


def analyze_liquid(image_path: Path, config: dict) -> str:
    gemini_cfg = config["gemini"]
    client = GeminiFallbackClient(
        api_keys=gemini_cfg["api_keys"],
        model_name=gemini_cfg.get("model_name", "models/gemini-2.5-flash"),
    )

    img = Image.open(image_path).convert("RGB")
    img.thumbnail((1024, 1024))

    prompt = (
        "Analyze this single image of a cup or glass. "
        "If NO cup or glass is visible, reply exactly: No cup detected. "
        "If visible, answer in one sentence using this format: "
        "There is a cup with a <Empty|Low|Medium|High> level of "
        "<Water|Coke|Fanta|Tea/Coffee|Milk|Cooking Oil|Roohafza|Unknown>."
    )

    result = client.generate_with_fallback(prompt, img)
    return result if result else "No cup detected."


def speak_text(text: str) -> None:
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Liquid Assist receiver + Gemini liquid level estimator")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to config JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    output_file = config["capture"].get("output_filename", "received_capture.jpg")
    image_path = ROOT / output_file

    print("Step 1/2: Waiting to receive photo from Raspberry Pi...")
    start = time.time()
    receive_image_tcp(config, image_path)
    print(f"Transfer completed in {time.time() - start:.2f}s")

    analysis = analyze_liquid(image_path, config)
    print("\nLiquid Assist result:")
    print(analysis)
    speak_text(analysis)


if __name__ == "__main__":
    main()
