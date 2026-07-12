from __future__ import annotations

import ipaddress
import difflib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import shutil
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


NUMBER_WORDS = {
    "one": "1",
    "first": "1",
    "scene": "1",
    "two": "2",
    "second": "2",
    "navigation": "2",
    "nav": "2",
    "three": "3",
    "third": "3",
    "hand": "3",
    "four": "4",
    "fourth": "4",
    "liquid": "4",
    "quit": "q",
    "exit": "q",
    "back": "m",
    "menu": "m",
    "switch": "m",
    "yes": "y",
    "yeah": "y",
    "no": "n",
}


PROJECT_ALIASES = {
    "1": (
        "1",
        "one",
        "first",
        "scene",
        "scene understanding",
        "understanding",
        "camera scene",
    ),
    "2": (
        "2",
        "two",
        "second",
        "nav",
        "navigation",
        "nav guidance",
        "navigation guidance",
        "guide me",
    ),
    "3": (
        "3",
        "three",
        "third",
        "hand",
        "hand guidance",
        "hand guide",
    ),
    "4": (
        "4",
        "four",
        "fourth",
        "liquid",
        "liquid assist",
        "liquid assistance",
        "water",
    ),
}


COMMAND_PHRASES = sorted(
    {
        "one",
        "option one",
        "scene",
        "scene understanding",
        "start scene",
        "start scene understanding",
        "open scene",
        "two",
        "option two",
        "nav",
        "navigation",
        "nav guidance",
        "navigation guidance",
        "start nav",
        "start navigation",
        "open navigation",
        "guide me",
        "three",
        "option three",
        "hand",
        "hand guidance",
        "start hand",
        "start hand guidance",
        "open hand guidance",
        "four",
        "option four",
        "liquid",
        "liquid assist",
        "liquid assistance",
        "start liquid",
        "start liquid assist",
        "open liquid assist",
        "switch mode",
        "change mode",
        "main menu",
        "go back",
        "back to menu",
        "stop task",
        "stop mode",
        "switch to scene",
        "switch to scene understanding",
        "switch to nav",
        "switch to navigation",
        "switch to nav guidance",
        "switch to hand",
        "switch to hand guidance",
        "switch to liquid",
        "switch to liquid assist",
        "quit",
        "exit",
        "close app",
        "yes",
        "yeah",
        "same",
        "correct",
        "no",
        "nope",
        "change",
    }
)


COMMON_MISHEARS = {
    "seen": "scene",
    "sin": "scene",
    "sen": "scene",
    "nave": "nav",
    "nab": "nav",
    "navigate": "navigation",
    "and guidance": "hand guidance",
    "hand guide": "hand guidance",
    "liquor": "liquid",
    "liquide": "liquid",
    "lequid": "liquid",
    "back menu": "back to menu",
    "stop current": "stop task",
}


class VoiceIO:
    def __init__(self) -> None:
        self.enabled = False
        self.speech_recognition = None
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.powershell = shutil.which("powershell") or shutil.which("powershell.exe")
        self.powershell_tts_available = False
        self.powershell_speech_available = False
        self.speech_output_warning_shown = False
        self.command_queue: queue.Queue[str] = queue.Queue()
        self.listen_stop_event = threading.Event()
        self.listen_thread: threading.Thread | None = None
        self.current_tts_process: subprocess.Popen | None = None
        self._tts_lock = threading.Lock()

    def setup(self) -> None:
        self._setup_tts()
        self._setup_speech_input()
        self._setup_windows_voice_fallback()

    def _setup_tts(self) -> None:
        try:
            import pyttsx3

            self.tts_engine = pyttsx3.init()
        except Exception:
            self.tts_engine = None

    def _setup_speech_input(self) -> None:
        try:
            import speech_recognition as sr

            self.speech_recognition = sr
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.7
            self.recognizer.non_speaking_duration = 0.4
            self.microphone = sr.Microphone()
        except Exception:
            self.speech_recognition = None
            self.recognizer = None
            self.microphone = None

    def _setup_windows_voice_fallback(self) -> None:
        if not self.powershell:
            return
        self.powershell_tts_available = self._powershell_check(
            "Add-Type -AssemblyName System.Speech; "
            "$null = New-Object System.Speech.Synthesis.SpeechSynthesizer"
        )
        self.powershell_speech_available = self._powershell_check(
            "Add-Type -AssemblyName System.Speech; "
            "$null = New-Object System.Speech.Recognition.SpeechRecognitionEngine"
        )

    def _powershell_check(self, script: str) -> bool:
        if not self.powershell:
            return False
        try:
            result = subprocess.run(
                [self.powershell, "-NoProfile", "-Command", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _powershell_say(self, text: str) -> bool:
        if not self.powershell or not self.powershell_tts_available:
            return False
        safe_text = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.Speak('{safe_text}')"
        )
        try:
            proc = subprocess.Popen(
                [self.powershell, "-NoProfile", "-Command", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.current_tts_process = proc
            proc.wait(timeout=max(8, min(30, len(text) // 8 + 8)))
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            self.stop_speaking()
            return False
        except Exception:
            return False
        finally:
            self.current_tts_process = None

    def _powershell_listen_once(self, timeout: int) -> str | None:
        if not self.powershell or not self.powershell_speech_available:
            return None
        quoted_phrases = ",".join(f"'{phrase.replace(chr(39), chr(39) + chr(39))}'" for phrase in COMMAND_PHRASES)
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
            "$recognizer.SetInputToDefaultAudioDevice(); "
            "$choices = New-Object System.Speech.Recognition.Choices; "
            f"$choices.Add([string[]]@({quoted_phrases})) | Out-Null; "
            "$builder = New-Object System.Speech.Recognition.GrammarBuilder; "
            "$builder.Culture = $recognizer.RecognizerInfo.Culture; "
            "$builder.Append($choices); "
            "$grammar = New-Object System.Speech.Recognition.Grammar($builder); "
            "$recognizer.LoadGrammar($grammar); "
            f"$result = $recognizer.Recognize([TimeSpan]::FromSeconds({timeout})); "
            "if ($result -ne $null -and $result.Confidence -ge 0.35) { $result.Text.ToLowerInvariant() }"
        )
        try:
            result = subprocess.run(
                [self.powershell, "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        text = result.stdout.strip().lower()
        return text or None

    @property
    def can_speak(self) -> bool:
        return self.tts_engine is not None or self.powershell_tts_available

    @property
    def can_listen(self) -> bool:
        return (
            self.recognizer is not None and self.microphone is not None
        ) or self.powershell_speech_available

    def describe_availability(self) -> None:
        print("\nVoice status:")
        print(f"  Spoken responses: {'on' if self.can_speak else 'not available'}")
        print(f"  Speech input: {'on' if self.can_listen else 'not available'}")
        print("  Keyboard input: always on")
        if not self.can_speak or not self.can_listen:
            print("  Tip: voice uses built-in Windows/Python options only; no admin setup is required.")

    def set_enabled(self, enabled: bool) -> None:
        if enabled and not (self.can_speak or self.can_listen):
            print("Voice could not start here, so keyboard input will be used.")
            self.enabled = False
            return
        self.enabled = enabled
        if self.enabled:
            print("Voice interaction enabled.")
            self.start_listening()
            self.say("Assistant is ready.")
        else:
            self.stop_listening()
            self.stop_speaking()
            print("Voice interaction disabled.")

    def start_listening(self) -> None:
        if not self.enabled or not self.can_listen:
            return
        if self.listen_thread is not None and self.listen_thread.is_alive():
            return
        self.listen_stop_event.clear()
        self.listen_thread = threading.Thread(target=self._listen_forever, daemon=True)
        self.listen_thread.start()

    def stop_listening(self) -> None:
        self.listen_stop_event.set()

    def _listen_forever(self) -> None:
        if self.recognizer is not None and self.microphone is not None:
            self._listen_forever_with_python()
            return
        self._listen_forever_with_windows()

    def _listen_forever_with_python(self) -> None:
        try:
            with self.microphone as source:
                try:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception:
                    pass
                while self.enabled and not self.listen_stop_event.is_set():
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        text = self._recognize_python_audio(audio)
                    except Exception:
                        continue
                    if text:
                        self.command_queue.put(text)
        except Exception:
            self._listen_forever_with_windows()

    def _recognize_python_audio(self, audio) -> str | None:
        if self.recognizer is None:
            return None
        for recognizer_name in ("recognize_sphinx", "recognize_google"):
            recognizer_func = getattr(self.recognizer, recognizer_name, None)
            if recognizer_func is None:
                continue
            try:
                text = recognizer_func(audio).strip().lower()
            except Exception:
                continue
            if text:
                return text
        return None

    def _listen_forever_with_windows(self) -> None:
        while self.enabled and not self.listen_stop_event.is_set():
            text = self._powershell_listen_once(timeout=2)
            if text:
                self.command_queue.put(text)
            else:
                time.sleep(0.1)

    def get_command(self, timeout: float = 0.0) -> str | None:
        if not self.enabled or not self.can_listen:
            return None
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_pending_commands(self) -> None:
        while True:
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break

    def stop_speaking(self) -> None:
        if self.tts_engine is not None:
            try:
                self.tts_engine.stop()
            except Exception:
                pass
        proc = self.current_tts_process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def say(self, text: str) -> None:
        if not self.enabled or not self.can_speak:
            return
        with self._tts_lock:
            if self.tts_engine is not None:
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    return
                except Exception:
                    self.tts_engine = None
            if not self._powershell_say(text):
                if not self.speech_output_warning_shown:
                    print("Speech output is unavailable right now; keyboard input is still available.")
                    self.speech_output_warning_shown = True
                self.powershell_tts_available = False

    def say_async(self, text: str) -> threading.Thread | None:
        if not self.enabled or not self.can_speak:
            return None
        self.stop_speaking()
        thread = threading.Thread(target=self.say, args=(text,), daemon=True)
        thread.start()
        return thread

    def listen_once(
        self,
        prompt: str | None = None,
        *,
        timeout: int = 5,
        phrase_time_limit: int = 5,
        quiet: bool = False,
    ) -> str | None:
        if not self.enabled or not self.can_listen:
            return None
        if prompt:
            if not quiet:
                print(prompt)
            self.say(prompt)
        if not quiet:
            print("Listening... You can also type your response.")
        if self.recognizer is not None and self.microphone is not None:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit,
                    )
                text = self.recognizer.recognize_google(audio).strip().lower()
                if text and not quiet:
                    print(f"Heard: {text}")
                return text
            except Exception:
                pass
        text = self._powershell_listen_once(timeout)
        if text and not quiet:
            print(f"Heard: {text}")
        if not text and not quiet:
            print("Could not understand speech. Please type your response.")
        return text


VOICE = VoiceIO()


def clean_command(raw_text: str) -> str:
    text = raw_text.strip().lower()
    text = re.sub(r"[^a-z0-9. ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    for wrong, fixed in COMMON_MISHEARS.items():
        text = text.replace(wrong, fixed)
    return text.strip()


def closest_command_phrase(command: str) -> str | None:
    if not command:
        return None
    matches = difflib.get_close_matches(command, COMMAND_PHRASES, n=1, cutoff=0.68)
    if matches:
        return matches[0]
    words = command.split()
    if len(words) == 1:
        all_words = sorted({word for phrase in COMMAND_PHRASES for word in phrase.split()})
        word_matches = difflib.get_close_matches(words[0], all_words, n=1, cutoff=0.72)
        if word_matches:
            return word_matches[0]
    return None


def parse_project_command(text: str) -> str | None:
    command = clean_command(text)
    if not command:
        return None
    words = set(command.split())
    matched_choices = set()
    for choice, aliases in PROJECT_ALIASES.items():
        for alias in aliases:
            alias_text = clean_command(alias)
            if alias_text == command:
                return choice
            if " " in alias_text and alias_text in command:
                matched_choices.add(choice)
            if alias_text in words:
                matched_choices.add(choice)
    if len(matched_choices) == 1:
        return next(iter(matched_choices))
    return None


def parse_assistant_command(raw_text: str) -> str:
    command = clean_command(raw_text)
    if not command:
        return ""

    project_choice = parse_project_command(command)
    if project_choice:
        return project_choice

    closest = closest_command_phrase(command)
    if closest and closest != command:
        project_choice = parse_project_command(closest)
        if project_choice:
            return project_choice
        command = closest

    if any(phrase in command for phrase in ("quit", "exit", "close app", "stop app")):
        return "q"
    if any(
        phrase in command
        for phrase in (
            "switch mode",
            "change mode",
            "main menu",
            "go back",
            "back to menu",
            "stop task",
            "stop mode",
        )
    ):
        return "m"
    if any(word in command.split() for word in ("yes", "yeah", "yep", "correct", "same")):
        return "y"
    if any(word in command.split() for word in ("no", "nope", "different", "change")):
        return "n"

    return command


def normalize_choice(raw_choice: str) -> str:
    choice = parse_assistant_command(raw_choice)
    if choice in PROJECTS or choice in {"q", "m", "y", "n"}:
        return choice
    for word, mapped in NUMBER_WORDS.items():
        if word in choice:
            return mapped
    return choice


def prompt_user(
    prompt: str,
    *,
    voice_prompt: str | None = None,
    speak_in_background: bool = False,
) -> str:
    spoken_prompt = voice_prompt or prompt
    if VOICE.enabled and speak_in_background:
        VOICE.say_async(spoken_prompt)
    elif VOICE.enabled:
        VOICE.say(spoken_prompt)
    if VOICE.enabled and VOICE.can_listen and os.name == "nt" and sys.stdin.isatty():
        try:
            return prompt_user_windows_voice_and_keyboard(prompt)
        except Exception:
            pass
    return input(prompt).strip()


def prompt_user_windows_voice_and_keyboard(prompt: str) -> str:
    import msvcrt

    print(prompt, end="", flush=True)
    typed_chars: list[str] = []
    while True:
        spoken = VOICE.get_command(timeout=0.05)
        if spoken:
            print(f"\nHeard: {spoken}")
            VOICE.stop_speaking()
            return spoken

        if msvcrt.kbhit():
            char = msvcrt.getwch()
            if char in {"\r", "\n"}:
                print()
                VOICE.stop_speaking()
                return "".join(typed_chars).strip()
            if char == "\b":
                if typed_chars:
                    typed_chars.pop()
                    print("\b \b", end="", flush=True)
                continue
            if char.isprintable():
                typed_chars.append(char)
                print(char, end="", flush=True)

        time.sleep(0.02)


def prompt_choice(prompt: str, valid_choices: set[str], invalid_message: str) -> str:
    while True:
        choice = normalize_choice(prompt_user(prompt))
        if choice in valid_choices:
            VOICE.stop_speaking()
            return choice
        print(invalid_message)
        VOICE.say(invalid_message)


def prompt_choice_while_speaking(
    prompt: str,
    valid_choices: set[str],
    invalid_message: str,
    announcement: str,
) -> str:
    while True:
        choice = normalize_choice(prompt_user(prompt, voice_prompt=announcement, speak_in_background=True))
        if choice in valid_choices:
            VOICE.stop_speaking()
            return choice
        print(invalid_message)
        VOICE.say(invalid_message)


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
    VOICE.clear_pending_commands()
    print("\nMain menu:")
    for key, project in PROJECTS.items():
        print(f"  {key}. {project['name']}")
    print("  q. Quit")
    print("During a mode: press M or say 'switch mode' to return here.")
    menu_announcement = (
        "Main menu. "
        "Option 1, Scene understanding. "
        "Option 2, Nav guidance. "
        "Option 3, Hand guidance. "
        "Option 4, Liquid assist. "
        "Say a mode name or number now. "
        "When a mode is running, say switch mode to come back here, or say switch to another mode. "
        "Say quit to close the launcher."
    )

    return prompt_choice_while_speaking(
        "\nEnter your choice: ",
        {*PROJECTS.keys(), "m", "q"},
        "Please choose Scene, Nav Guidance, Hand Guidance, Liquid Assist, or Quit.",
        menu_announcement,
    )


def ask_ip_address() -> str:
    while True:
        ip_text = prompt_user(
            "\nEnter IP address (example: 192.168.1.25): ",
            voice_prompt="Say or type the Raspberry Pi IP address.",
        )
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
    choice = prompt_choice(
        "Is this the same rasberry pi IP as before? (y/n): ",
        {"y", "n"},
        "Invalid option. Please type y or n.",
    )
    return choice == "y"


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


def configure_ip_if_needed(*, force: bool = False) -> None:
    current_ip = get_current_ip()
    if current_ip is None:
        print("\nCould not detect an existing IP in config files.")
        ip_value = ask_ip_address()
        update_config_files(ip_value)
        return

    if force:
        print(f"\nCurrent saved IP of rasberry pi is: {current_ip}")
        ip_value = ask_ip_address()
        update_config_files(ip_value)
        return

    keep_same = ask_if_ip_same(current_ip)
    if not keep_same:
        ip_value = ask_ip_address()
        update_config_files(ip_value)
    else:
        print("Keeping existing IP. No config changes made.")


def switch_target_from_command(text: str | None) -> str | None:
    if not text:
        return None
    command = normalize_choice(text)
    if command in PROJECTS or command == "m":
        return command
    return None


def start_voice_switch_listener(
    stop_event: threading.Event,
    switch_event: threading.Event,
    target_queue: queue.Queue[str | None],
) -> threading.Thread | None:
    if not VOICE.enabled or not VOICE.can_listen:
        return None

    def listen_for_switch() -> None:
        while not stop_event.is_set() and not switch_event.is_set():
            heard = VOICE.get_command(timeout=0.2)
            target = switch_target_from_command(heard)
            if target:
                VOICE.stop_speaking()
                target_queue.put(None if target == "m" else target)
                switch_event.set()
                break

    thread = threading.Thread(target=listen_for_switch, daemon=True)
    thread.start()
    return thread


def poll_keyboard_for_switch(
    proc: subprocess.Popen,
    switch_event: threading.Event,
    target_queue: queue.Queue[str | None],
) -> None:
    if os.name != "nt":
        return
    try:
        import msvcrt
    except ImportError:
        return

    while proc.poll() is None and not switch_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            if key in PROJECTS:
                VOICE.stop_speaking()
                target_queue.put(key)
                switch_event.set()
                break
            if key == "m":
                VOICE.stop_speaking()
                target_queue.put(None)
                switch_event.set()
                break
        time.sleep(0.1)


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    print("\nStopping current task and returning to the main menu...")
    VOICE.say("Stopping current task and returning to the main menu.")
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_project(choice: str) -> tuple[int | None, bool, str | None]:
    project = PROJECTS[choice]
    venv_python = project["venv_python"]
    python_exe = str(venv_python if venv_python.exists() else sys.executable)
    cmd = [python_exe, *project["command"]]

    print(f"\nRunning {project['name']}...")
    print("Command:", " ".join(cmd))
    print(f"Working directory: {project['folder']}")
    print("Assistant controls while running:")
    print("  Press M or say 'switch mode' to stop this task and return to the main menu.")
    print("  Press 1, 2, 3, or 4, or say 'switch to nav/hand/scene/liquid' to change directly.")
    if VOICE.enabled and VOICE.can_listen:
        print('  Voice examples: "switch to nav", "start hand guidance", "main menu", "stop task".')
    VOICE.say(
        f"Starting {project['name']}. "
        "You can say switch mode to return to the menu, or say switch to another mode."
    )

    proc = subprocess.Popen(cmd, cwd=project["folder"])
    switch_event = threading.Event()
    voice_stop_event = threading.Event()
    target_queue: queue.Queue[str | None] = queue.Queue()
    voice_thread = start_voice_switch_listener(voice_stop_event, switch_event, target_queue)

    keyboard_thread = threading.Thread(
        target=poll_keyboard_for_switch,
        args=(proc, switch_event, target_queue),
        daemon=True,
    )
    keyboard_thread.start()

    try:
        while proc.poll() is None:
            if switch_event.is_set():
                stop_process(proc)
                try:
                    next_choice = target_queue.get_nowait()
                except queue.Empty:
                    next_choice = None
                return None, True, next_choice
            time.sleep(0.2)
        return proc.returncode, False, None
    finally:
        voice_stop_event.set()
        if voice_thread is not None:
            voice_thread.join(timeout=1)


def toggle_voice() -> None:
    VOICE.describe_availability()
    if VOICE.enabled:
        VOICE.set_enabled(False)
        return
    choice = prompt_choice(
        "Enable voice interaction? (y/n): ",
        {"y", "n"},
        "Invalid option. Please type y or n.",
    )
    VOICE.set_enabled(choice == "y")


def main() -> None:
    check_venvs()
    configure_ip_if_needed()
    VOICE.setup()
    VOICE.describe_availability()
    VOICE.set_enabled(VOICE.can_speak or VOICE.can_listen)

    next_choice: str | None = None
    while True:
        choice = next_choice or choose_project()
        next_choice = None

        if choice == "q":
            print("Exited.")
            VOICE.say("Exited.")
            return
        if choice == "m":
            print("Already at the main menu. Choose a mode to start.")
            VOICE.say("Already at the main menu. Choose a mode to start.")
            continue
        exit_code, switched, switch_target = run_project(choice)
        if switched:
            if switch_target in PROJECTS:
                next_choice = switch_target
            continue
        print(f"\nProcess finished with exit code: {exit_code}")
        VOICE.say(f"Process finished with exit code {exit_code}.")


if __name__ == "__main__":
    main()
