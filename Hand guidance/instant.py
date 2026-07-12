
import cv2

cap = cv2.VideoCapture("udp://192.168.114.5:5000", cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("ERROR: Cannot open stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received")
        break

    cv2.imshow("RPi Live Stream", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Rasberry pi command to run 
# rpicam-vid -t 0 \
# --width 640 --height 480 \
# --framerate 30 \
# --codec h264 \
# --inline \
# -o udp://192.168.114.114:5000
