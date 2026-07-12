import argparse
import os
import sys
import time

import cv2

from model_inference import ModelInference
from record import get_latest_recording, receive_video
from scene_interpreter import SceneInterpreter
from speech_output import SpeechOutput
from video_processor import VideoProcessor


DEFAULT_MODE = os.getenv("MODE", "gemini")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Real-time Environmental Awareness for visually impaired users"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: path to video file or webcam. Defaults to latest recording if present, otherwise webcam.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["gemini"],
        help="Model mode. Only gemini is supported.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Frame sampling interval in seconds.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the video feed in a window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting.",
    )
    parser.add_argument(
        "--google-api-key",
        type=str,
        default=None,
        help="Google Gemini API key for gemini mode. Overrides GOOGLE_API_KEY environment variable.",
    )
    parser.add_argument(
        "--receive-first",
        action="store_true",
        help="Receive video first, then process that received file in the same run.",
    )
    return parser


def resolve_source(source_arg):
    if source_arg:
        return source_arg if source_arg.lower() != "webcam" else "webcam"
    latest_recording = get_latest_recording()
    if latest_recording:
        return latest_recording
    return "webcam"


def main():
    parser = build_parser()
    args = parser.parse_args()
    source = receive_video() if args.receive_first else resolve_source(args.source)
    mode = args.mode

    print(f"Starting Environmental Awareness: source={source}")

    try:
        processor = VideoProcessor(source=source, sample_interval=args.interval, max_frames=args.max_frames)
        processor.open()
    except Exception as exc:
        print(f"ERROR: Failed to open source: {exc}")
        sys.exit(1)

    inferencer = ModelInference(mode=mode, google_api_key=args.google_api_key)
    interpreter = SceneInterpreter()
    speaker = SpeechOutput()

    try:
        for frame, timestamp in processor.frame_generator():
            try:
                raw_text = inferencer.infer(frame)
            except Exception as exc:
                print(f"Inference error: {exc}")
                continue

            messages = interpreter.interpret(raw_text)
            if messages:
                for message in messages:
                    print(f"[{timestamp:.1f}s] {message}")
                    speaker.say(message)
            else:
                print(f"[{timestamp:.1f}s] No meaningful change detected.")

            if args.display:
                try:
                    overlay = frame.copy()
                    cv2.putText(
                        overlay,
                        f"Mode: {mode}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Environmental Awareness", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except Exception:
                    pass
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        processor.release()
        speaker.shutdown()
        if args.display:
            cv2.destroyAllWindows()
        print("Shutting down.")


if __name__ == "__main__":
    main()
