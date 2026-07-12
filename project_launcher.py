from __future__ import annotations

import ipaddress
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HAND_CONFIG_PATH = ROOT / "Hand guidance" / "config.json"
NAV_CONFIG_PATH = ROOT / "Nav guidance" / "config.yaml"
LIQUID_CONFIG_PATH = ROOT / "Liquid Assist" / "config.json"

PROJECTS = {
    "1": {
        "name": "Scene",
        "folder": ROOT / "scene",
        "venv_python": ROOT / "scene" / "venv" / "Scripts" / "python.exe",
        "command": ["main.py", "--receive-first"],
    },
    "2": {
        "name": "Nav Guidance",
        "folder": ROOT / "Nav guidance",
        "venv_python": ROOT / "Nav guidance" / "venv" / "Scripts" / "python.exe",
        "command": ["run_assistive_system.py", "--config", "config.yaml"],
    },
    "3": {
        "name": "Hand Guidance",
        "folder": ROOT / "Hand guidance",
        "venv_python": ROOT / "Hand guidance" / "venv" / "Scripts" / "python.exe",
        "command": ["vision_assistant.py"],
    },
    "4": {
        "name": "Liquid Assist",
        "folder": ROOT / "Liquid Assist",
        "venv_python": ROOT / "Liquid Assist" / "venv" / "Scripts" / "python.exe",
        "command": ["main.py"],
        "allow_system_python": True,
    },
}


def check_venvs() -> None:
    print("Checking virtual environments...")
    missing = []
    for key, project in PROJECTS.items():
        py = project["venv_python"]
        if py.exists():
            print(f"  [{key}] {project['name']}: OK -> {py}")
        elif project.get("allow_system_python"):
            print(f"  [{key}] {project['name']}: venv missing, fallback to current Python -> {sys.executable}")
        else:
            print(f"  [{key}] {project['name']}: MISSING -> {py}")
            missing.append(project["name"])

    if missing:
        print("\nOne or more virtual environments are missing. Fix them and try again.")
        sys.exit(1)


def choose_project() -> str:
    print("\nChoose a project to run:")
    for key, project in PROJECTS.items():
        print(f"  {key}. {project['name']}")
    print("  q. Quit")

    while True:
        choice = input("\nEnter your choice: ").strip().lower()
        if choice in PROJECTS or choice == "q":
            return choice
        print("Invalid option. Please choose 1, 2, 3, 4, or q.")


def ask_ip_address() -> str:
    while True:
        ip_text = input("\nEnter IP address (example: 192.168.1.25): ").strip()
        try:
            ipaddress.ip_address(ip_text)
            return ip_text
        except ValueError:
            print("Invalid IP format. Please enter a valid IPv4/IPv6 address.")


def update_udp_ip_in_text(text: str, new_ip: str) -> str:
    pattern = r"udp://([^:/\"?]+):5000"
    return re.sub(pattern, f"udp://{new_ip}:5000", text)


def extract_first_ip_from_text(text: str) -> str | None:
    match = re.search(r"udp://([^:/\"?]+):5000", text)
    if not match:
        return None
    return match.group(1)


def get_current_ip() -> str | None:
    hand_text = HAND_CONFIG_PATH.read_text(encoding="utf-8")
    nav_text = NAV_CONFIG_PATH.read_text(encoding="utf-8")
    liquid_ip = None
    if LIQUID_CONFIG_PATH.exists():
        liquid_data = json.loads(LIQUID_CONFIG_PATH.read_text(encoding="utf-8"))
        liquid_ip = liquid_data.get("network", {}).get("raspberry_pi_ip")
    return extract_first_ip_from_text(hand_text) or extract_first_ip_from_text(nav_text) or liquid_ip


def ask_if_ip_same(current_ip: str) -> bool:
    print(f"\nCurrent saved IP of rasberry pi is: {current_ip}")
    while True:
        choice = input("Is this the same rasberry pi IP as before? (y/n): ").strip().lower()
        if choice in {"y", "n"}:
            return choice == "y"
        print("Invalid option. Please type y or n.")


def update_config_files(new_ip: str) -> None:
    hand_original = HAND_CONFIG_PATH.read_text(encoding="utf-8")
    nav_original = NAV_CONFIG_PATH.read_text(encoding="utf-8")

    hand_updated = update_udp_ip_in_text(hand_original, new_ip)
    nav_updated = update_udp_ip_in_text(nav_original, new_ip)

    HAND_CONFIG_PATH.write_text(hand_updated, encoding="utf-8")
    NAV_CONFIG_PATH.write_text(nav_updated, encoding="utf-8")

    updated_paths = [HAND_CONFIG_PATH, NAV_CONFIG_PATH]

    if LIQUID_CONFIG_PATH.exists():
        liquid_data = json.loads(LIQUID_CONFIG_PATH.read_text(encoding="utf-8"))
        liquid_data.setdefault("network", {})["raspberry_pi_ip"] = new_ip
        LIQUID_CONFIG_PATH.write_text(json.dumps(liquid_data, indent=2), encoding="utf-8")
        updated_paths.append(LIQUID_CONFIG_PATH)

    print("\nUpdated IP in:")
    for path in updated_paths:
        print(f"  - {path}")


def run_project(choice: str) -> int:
    project = PROJECTS[choice]
    venv_python = project["venv_python"]
    python_exe = str(venv_python if venv_python.exists() else sys.executable)
    cmd = [python_exe, *project["command"]]

    print(f"\nRunning {project['name']}...")
    print("Command:", " ".join(cmd))
    print(f"Working directory: {project['folder']}")

    return subprocess.call(cmd, cwd=project["folder"])


def main() -> None:
    check_venvs()
    current_ip = get_current_ip()
    if current_ip is None:
        print("\nCould not detect an existing IP in config files.")
        ip_value = ask_ip_address()
        update_config_files(ip_value)
    else:
        keep_same = ask_if_ip_same(current_ip)
        if not keep_same:
            ip_value = ask_ip_address()
            update_config_files(ip_value)
        else:
            print("Keeping existing IP. No config changes made.")

    choice = choose_project()

    if choice == "q":
        print("Exited.")
        return

    exit_code = run_project(choice)
    print(f"\nProcess finished with exit code: {exit_code}")


if __name__ == "__main__":
    main()
