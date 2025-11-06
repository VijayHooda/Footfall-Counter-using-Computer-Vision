"""
🧠 AI Assignment — Footfall Counter using Computer Vision
Author: Vijay Hooda
Description:
Detects and counts people entering and exiting using YOLOv8 + SORT tracker.
Processes a given video ( mall CCTV ) and saves the analyzed output video.
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Make sure you have the pure Python sort.py file in the same folder

# -------------------------------
# ⚙️ CONFIGURATION
# -------------------------------
VIDEO_PATH = "test_video.mp4"  # path to your downloaded input video
OUTPUT_PATH = "output_processed.mp4"  # path for saving output
MODEL_PATH = "yolov8n.pt"  # lightweight YOLO model

# -------------------------------
# 🚀 INITIALIZE COMPONENTS
# -------------------------------
print("🚀 Running Footfall Counter...")

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Initialize SORT tracker
tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.2)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video file: {VIDEO_PATH}")

# Prepare video writer (no imshow)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    OUTPUT_PATH, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))
)

# Counting variables
total_in = 0
total_out = 0
counted_ids = set()

# Define the reference line (middle of frame)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
line_y = frame_height // 2

# -------------------------------
# 🧮 PROCESS VIDEO FRAMES
# -------------------------------
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    detections = []

    # Run YOLOv8 inference (only detect persons)
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            if cls == 0:  # class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                detections.append([x1, y1, x2, y2, conf])

    dets = np.array(detections)
    if len(dets) == 0:
        dets = np.empty((0, 5))

    # Update tracker
    tracked_objects = tracker.update(dets)

    # Draw everything and count crossings
    for x1, y1, x2, y2, obj_id in tracked_objects:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        color = (0, 255, 0)

        if obj_id not in counted_ids:
            if cy < line_y - 10:
                direction = "UP"
            elif cy > line_y + 10:
                direction = "DOWN"
            else:
                direction = None

            if direction == "UP":
                total_out += 1
                counted_ids.add(obj_id)
            elif direction == "DOWN":
                total_in += 1
                counted_ids.add(obj_id)

        # Draw bounding box and ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"ID {int(obj_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Draw the counting line and stats
    cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 0), 2)
    cv2.putText(frame, f"IN: {total_in}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {total_out}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save frame to output file
    out.write(frame)

# -------------------------------
# 🏁 CLEANUP
# -------------------------------
cap.release()
out.release()

end_time = time.time()
total_time = end_time - start_time
print(f"\n✅ Footfall counting complete!")
print(f"🕒 Processed {frame_count} frames in {total_time:.2f} sec "
      f"({frame_count / total_time:.1f} FPS)")
print(f"📊 Final Counts → IN: {total_in}, OUT: {total_out}")
print(f"💾 Output saved to: {OUTPUT_PATH}")
