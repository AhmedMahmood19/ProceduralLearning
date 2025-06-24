# Check that the videos will be read correctly by OpenCV, after you run the shell script to re-encode the videos 
import cv2

videopath="/workspace/Egoprocel/videos/Epic-Tents/02.tent.120617.gopro.MP4"

cap = cv2.VideoCapture(videopath)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps if fps > 0 else "unknown"

print(f"FPS: {fps}")
print(f"Frame count: {frame_count}")
print(f"Duration (sec): {duration}")

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Stopped reading at frame {frame_num}")
        break
    frame_num += 1
