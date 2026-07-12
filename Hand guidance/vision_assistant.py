import cv2
import numpy as np
import subprocess
import threading
import time
import queue
import os
import json
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
DEFAULT_STREAM_URL = "udp://192.168.114.5:5000?fifo_size=50000000&overrun_nonfatal=1"

def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not load config.json ({exc}). Using default settings.")
        return {}

# 0. Bufferless Video Capture for Real-Time UDP
class BufferlessVideoCapture:
    def __init__(self, name):
        # Force low-latency FFMPEG options
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "probesize;32|analyzeduration;0|sync;ext|thread_type;slice|fflags;nobuffer|flags;low_delay"
        self.cap = cv2.VideoCapture(name, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue()
        self.running = True
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait() # Discard old frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return True, self.q.get()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()

# 1. Bulletproof Windows Speech (Native PowerShell)
def speak_native(text):
    def _run():
        # Using PowerShell for speech avoids all COM/threading hangs in Python
        ps_command = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
        subprocess.run(["powershell", "-Command", ps_command], capture_output=True)
    threading.Thread(target=_run, daemon=True).start()

# 2. Heavy-Duty Background Object Detection
class ObjectDetector(threading.Thread):
    def __init__(self, target_name):
        super().__init__(daemon=True)
        # Upgraded to the ultra-powerful YOLOv8 Extra-Large World model
        # This is an open-vocabulary model that can detect 'anything' accurately.
        self.model = YOLO('yolov8x-worldv2.pt')
        
        # Automatic GPU acceleration if available
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to('cuda')
                print("--- GPU ACCELERATION ENABLED ---")
        except:
            pass

        # Set classes dynamically - No more manual mapping needed!
        self.target_name = target_name.lower().strip()
        self.model.set_classes([self.target_name])
        
        self.frame = None
        self.result_box = None
        self.running = True

    def update_frame(self, frame):
        self.frame = frame

    def run(self):
        while self.running:
            if self.frame is not None:
                # High resolution and high precision inference
                # The 'World' model will only look for the classes we set above.
                results = self.model(self.frame, verbose=False, imgsz=640, conf=0.1)
                max_area = 0
                best_box = None
                
                for r in results:
                    for box in r.boxes:
                        # Since we set exactly one class, every detection is our target
                        b = box.xyxy[0].cpu().numpy()
                        area = (b[2] - b[0]) * (b[3] - b[1])
                        if area > max_area:
                            max_area = area
                            best_box = b
                self.result_box = best_box
            time.sleep(0.01)

def get_guidance(hand_pos, target_pos, frame_size):
    hx, hy = hand_pos
    tx, ty = target_pos
    w, h = frame_size
    tx_tol, ty_tol = w * 0.12, h * 0.12
    
    instr = []
    if hx < tx - tx_tol: instr.append("Right")
    elif hx > tx + tx_tol: instr.append("Left")
    
    if hy < ty - ty_tol: instr.append("Lower")
    elif hy > ty + ty_tol: instr.append("Raise")
    
    dist = np.sqrt((hx - tx)**2 + (hy - ty)**2)
    if dist > w * 0.15: instr.append("Forward")
    
    return " ".join(instr) if instr else "Stop. Reach now."

def main():
    print("\n--- POWERFUL VISION ASSISTANT ---")
    config = load_config()
    target_object = input("Enter object (e.g., mouse, cup, bottle, pen): ").lower().strip()
    
    speak_native(f"Searching for {target_object}. Point the camera and show your hand.")
    
    # Hand Tracking Setup
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)

    detector = ObjectDetector(target_object)
    detector.start()

    # --- Robust Stream Handling (Optimized for Ultra-Low Latency) ---
    stream_url = config.get("stream_url", DEFAULT_STREAM_URL)
    
    # Try the Bufferless Stream first
    try:
        print(f"Connecting to {stream_url}...")
        cap = BufferlessVideoCapture(stream_url)
        if not cap.isOpened():
            raise Exception("Stream not reachable")
    except:
        print("WARNING: UDP Stream failed. Falling back to local camera (0)...")
        cap = BufferlessVideoCapture(0)

    last_speak_time = 0
    cooldown = 1.2

    while True:
        success, frame = cap.read()
        if not success:
            print("Frame dropped. Attempting to reconnect...")
            time.sleep(0.5)
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        detector.update_frame(frame)
        
        # 1. Hand Tracking
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_results = landmarker.detect(mp_image)
        
        hand_pos = None
        if hand_results.hand_landmarks:
            lms = hand_results.hand_landmarks[0]
            hand_pos = (int(np.mean([m.x for m in lms]) * w), int(np.mean([m.y for m in lms]) * h))
            cv2.circle(frame, hand_pos, 15, (255, 0, 0), -1)

        # 2. Logic & Guidance
        target_box = detector.result_box
        if target_box is not None:
            tx1, ty1, tx2, ty2 = target_box
            tx_center, ty_center = int((tx1 + tx2) / 2), int((ty1 + ty2) / 2)
            
            cv2.rectangle(frame, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (0, 255, 0), 2)
            
            if hand_pos:
                cv2.line(frame, hand_pos, (tx_center, ty_center), (255, 255, 0), 2)
                guidance = get_guidance(hand_pos, (tx_center, ty_center), (w, h))
                
                if time.time() - last_speak_time > cooldown:
                    speak_native(guidance)
                    last_speak_time = time.time()
                
                cv2.putText(frame, f"ACTION: {guidance}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            else:
                cv2.putText(frame, "WAITING FOR HAND", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"FINDING {target_object.upper()}...", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Vision Assistant PRO", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    detector.running = False
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
