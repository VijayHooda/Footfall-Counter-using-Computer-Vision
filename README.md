# 🧠 Footfall Counter using Computer Vision

## 📍 Objective
To develop a computer vision system that detects, tracks, and counts people entering and exiting a region (e.g., a mall gate or corridor) in a video using deep learning–based detection and object tracking.

---

## 🚀 Approach
The implemented system follows these steps:

1. **Detection (YOLOv8):**  
   Each frame of the video is processed using the Ultralytics YOLOv8 model, which identifies all persons present (`class=0`).

2. **Tracking (SORT Algorithm):**  
   The [SORT (Simple Online and Realtime Tracking)](https://github.com/abewley/sort) algorithm assigns unique IDs to each detected person and maintains their identity across frames using a Kalman Filter and IoU-based data association.

3. **Counting Logic:**  
   A virtual **horizontal line** is drawn across the middle of the frame.  
   - If a person’s centroid moves **downward across the line**, it counts as an **“IN”**.  
   - If a person’s centroid moves **upward across the line**, it counts as an **“OUT”**.  
   Each person is counted only once using their unique tracking ID.

4. **Output:**  
   Instead of displaying frames (to avoid GUI errors in headless environments), the processed output video is **saved as** `output_processed.mp4`, containing bounding boxes, IDs, and live IN/OUT counts.

---

## 🎥 Video Source
**Title:** *People Entering and Exiting Mall Stock Footage*  
**Source:** [YouTube - https://www.youtube.com/watch?v=gAuJlwnUqMs](https://www.youtube.com/watch?v=gAuJlwnUqMs)

The downloaded video was renamed as `test_video.mp4` and used as the input dataset for analysis.

---

## 🧩 Tools & Libraries
- Python ≥ 3.8  
- OpenCV (for frame handling and video I/O)  
- Ultralytics YOLOv8 (for object detection)  
- SORT (pure Python implementation for object tracking)  
- NumPy (for array operations)  
- FilterPy (for Kalman filtering within SORT)  
- Lapx (for efficient linear assignment)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/VijayHooda/Footfall-Counter-using-Computer-Vision.git
cd Footfall-Counter-using-Computer-Vision
