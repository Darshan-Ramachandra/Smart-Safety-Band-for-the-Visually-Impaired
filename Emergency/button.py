import socket
import os
import json
from contextlib import contextmanager
import re
import subprocess
import sys
import time
import threading
import urllib.request
import urllib.error

PORT = 5000
TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "8790379442:AAFnemkTY3CIv3p8KTEchuCwlAfcnZ5ERF4",
)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1362217329")
TRIGGER_TEXTS = {"SOS", "BUTTON_PRESSED"}
PHONE_IP = os.getenv("PHONE_IP", "100.66.31.220")
PHONE_PORT = int(os.getenv("PHONE_PORT", "8080"))
TELEGRAM_API_BASE = os.getenv("TELEGRAM_API_BASE", "https://api.telegram.org")
TELEGRAM_FORCE_IPV4 = os.getenv("TELEGRAM_FORCE_IPV4", "1").lower() not in ("0", "false", "no")
TELEGRAM_USE_SYSTEM_PROXY = os.getenv("TELEGRAM_USE_SYSTEM_PROXY", "0").lower() in (
    "1",
    "true",
    "yes",
)


def _nmea_to_decimal(raw_value: str, direction: str, is_lat: bool) -> float:
    degree_digits = 2 if is_lat else 3
    degrees = float(raw_value[:degree_digits])
    minutes = float(raw_value[degree_digits:])
    decimal = degrees + (minutes / 60.0)
    if direction in ("S", "W"):
        decimal = -decimal
    return decimal


def _extract_location_from_text(text: str):
    decimal_pair_match = re.search(
        r"(-?\d{1,2}\.\d{4,})\s*,\s*(-?\d{1,3}\.\d{4,})",
        text,
    )
    if decimal_pair_match:
        try:
            return float(decimal_pair_match.group(1)), float(decimal_pair_match.group(2))
        except Exception:
            pass

    for rmc_match in re.finditer(
        r"\$(?:GP|GN)RMC,[^,\r\n]*,A,(\d{4,5}\.\d+),([NS]),(\d{5,6}\.\d+),([EW])",
        text,
    ):
        try:
            lat = _nmea_to_decimal(rmc_match.group(1), rmc_match.group(2), is_lat=True)
            lon = _nmea_to_decimal(rmc_match.group(3), rmc_match.group(4), is_lat=False)
            return lat, lon
        except Exception:
            continue

    for gga_match in re.finditer(
        r"\$(?:GP|GN)GGA,[^,\r\n]*,(\d{4,5}\.\d+),([NS]),(\d{5,6}\.\d+),([EW]),([1-9])",
        text,
    ):
        try:
            lat = _nmea_to_decimal(gga_match.group(1), gga_match.group(2), is_lat=True)
            lon = _nmea_to_decimal(gga_match.group(3), gga_match.group(4), is_lat=False)
            return lat, lon
        except Exception:
            continue

    for line in text.splitlines():
        gps_fix_match = re.search(
            r"GPS\s*fix\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
            line,
            re.IGNORECASE,
        )
        if gps_fix_match:
            try:
                return float(gps_fix_match.group(1)), float(gps_fix_match.group(2))
            except Exception:
                pass

        parts = line.strip().split(",")
        if not parts:
            continue

        if parts[0] in ("$GPRMC", "$GNRMC") and len(parts) >= 7 and parts[2] == "A":
            try:
                lat = _nmea_to_decimal(parts[3], parts[4], is_lat=True)
                lon = _nmea_to_decimal(parts[5], parts[6], is_lat=False)
                return lat, lon
            except Exception:
                continue

        if parts[0] in ("$GPGGA", "$GNGGA") and len(parts) >= 7 and parts[6] not in ("0", ""):
            try:
                lat = _nmea_to_decimal(parts[2], parts[3], is_lat=True)
                lon = _nmea_to_decimal(parts[4], parts[5], is_lat=False)
                return lat, lon
            except Exception:
                continue

    return None


def get_location_direct_from_phone(timeout_seconds: int = 30):
    end_time = time.time() + timeout_seconds
    buffer = ""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(2)

    try:
        client_socket.connect((PHONE_IP, PHONE_PORT))
        chunk_count = 0
        while time.time() < end_time:
            try:
                data = client_socket.recv(1024)
            except socket.timeout:
                continue

            if not data:
                break

            chunk_count += 1
            buffer += data.decode("utf-8", errors="ignore")
            location = _extract_location_from_text(buffer)
            if location:
                return location

        if buffer.strip():
            print(f"Direct GPS read received {chunk_count} chunks but no parseable location.")
            print(buffer[:1000])
        else:
            print("Direct GPS read got no data from phone socket.")

        return None
    except Exception as exc:
        print(f"Direct GPS socket read failed: {exc}")
        return None
    finally:
        client_socket.close()


def get_location_by_running_gps_py(timeout_seconds: int = 20):
    gps_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gps.py")
    proc = subprocess.Popen(
        [sys.executable, "-u", gps_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
    )
    output_chunks = []

    def _reader():
        if proc.stdout is None:
            return
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            output_chunks.append(chunk)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    end_time = time.time() + timeout_seconds

    try:
        while time.time() < end_time:
            if output_chunks:
                combined = b"".join(output_chunks).decode("utf-8", errors="ignore")
                location = _extract_location_from_text(combined)
                if location:
                    return location

            if proc.poll() is not None:
                break
            time.sleep(0.2)

        combined = b"".join(output_chunks).decode("utf-8", errors="ignore")
        location = _extract_location_from_text(combined)
        if location:
            return location

        if combined.strip():
            print("gps.py output captured but no coordinates parsed:")
            print(repr(combined[:1000]))
        else:
            print("gps.py produced no output within timeout window.")

        return None
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        reader_thread.join(timeout=1)


def _telegram_error_details(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        body = exc.read().decode("utf-8", errors="ignore")
        return f"HTTP {exc.code}: {body}"

    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)

    return str(exc)


@contextmanager
def _prefer_ipv4_for_telegram():
    if not TELEGRAM_FORCE_IPV4:
        yield
        return

    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
        results = original_getaddrinfo(host, port, family, type, proto, flags)
        ipv4_results = [result for result in results if result[0] == socket.AF_INET]
        other_results = [result for result in results if result[0] != socket.AF_INET]
        return ipv4_results + other_results

    socket.getaddrinfo = getaddrinfo_ipv4_first
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def send_telegram_message(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram config missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return False

    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode("utf-8")
    url = f"{TELEGRAM_API_BASE}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "EmergencySOS/1.0",
        },
        method="POST",
    )
    if TELEGRAM_USE_SYSTEM_PROXY:
        opener = urllib.request.build_opener()
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    last_error = None

    for attempt in range(1, 4):
        try:
            with _prefer_ipv4_for_telegram():
                response = opener.open(request, timeout=15)
            with response:
                body = response.read().decode("utf-8", errors="ignore")
                if '"ok":true' in body:
                    return True
                print(f"Telegram API response: {body}")
                last_error = body
        except Exception as exc:
            last_error = _telegram_error_details(exc)
            print(f"Telegram send attempt {attempt} failed: {last_error}")

        time.sleep(1)

    print(
        "Telegram send failed after retries. If the error says the connection was "
        "forcibly closed/reset, this PC or network is blocking the HTTPS connection "
        "to api.telegram.org. IPv4 is preferred by default because this network's "
        "Telegram IPv6 route may fail. System proxies are bypassed by default; set "
        "TELEGRAM_USE_SYSTEM_PROXY=1 only if your network requires a proxy. You can "
        "also try a mobile hotspot/VPN or allow Python in firewall/antivirus."
    )
    if last_error:
        print(f"Last Telegram error: {last_error}")

    return False

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", PORT))  # Listen on all interfaces
server.listen(1)

print(f"Waiting for Raspberry Pi to connect on port {PORT}...")
conn, addr = server.accept()
print(f"Connected from {addr}")

try:
    while True:
        data = conn.recv(1024)
        if not data:
            print("Pi disconnected")
            break

        message = data.decode(errors="ignore").strip()
        print(f"Received: {message}")

        if message.upper() in TRIGGER_TEXTS:
            print("SOS received. Fetching GPS location...")
            location = get_location_direct_from_phone(timeout_seconds=35)
            if location is None:
                print("Direct read failed, trying gps.py fallback...")
                location = get_location_by_running_gps_py(timeout_seconds=45)

            if location is None:
                send_telegram_message(
                    "Emergency alert triggered, but GPS location could not be fetched."
                )
                print("GPS fix not available.")
                continue

            lat, lon = location
            maps_link = f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"
            telegram_text = (
                "Emergency alert!\n"
                f"Live location: {lat:.6f}, {lon:.6f}\n"
                f"{maps_link}"
            )

            if send_telegram_message(telegram_text):
                print("Location sent to Telegram.")
            else:
                print("Could not send location to Telegram.")

except KeyboardInterrupt:
    print("Stopped")

finally:
    conn.close()
    server.close()
