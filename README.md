# 🧠 Footfall Counter using Computer Vision

## 📍 Objective
To build a computer vision-based system that detects, tracks, and counts people entering and exiting a region (doorway, corridor, or gate) in a video using deep learning.

---

## 🚀 Approach
The system uses:
- **YOLOv8** for person detection.
- **SORT** tracker for maintaining identities across frames.
- A **virtual horizontal line** as the Region of Interest (ROI).
- Counts people crossing the line in both directions.

--- Video Link - https://www.youtube.com/watch?v=gAuJlwnUqMs

## 🧩 Tools & Libraries
- Python ≥ 3.8  
- OpenCV  
- Ultralytics YOLOv8  
- SORT Tracker (`pip install sort-tracker`)  
- NumPy  

---

## ⚙️ Setup Instructions
```bash
# Clone this repository
git clone https://github.com/<your-username>/Footfall-Counter-using-Computer-Vision.git
cd Footfall-Counter-using-Computer-Vision

# Install dependencies
pip install -r requirements.txt
