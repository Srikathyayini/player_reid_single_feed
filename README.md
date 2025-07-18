# Player Re-Identification (Single Camera Feed)

## Objective
Track players in a 15s football clip and re-identify them when they re-enter the frame.

## Approach
- YOLOv11 used for object detection (class `0` as player).
- Players tracked using position-based heuristics.
- Frame-wise ID assignment and display.
- Output visualized in real time.

## Dependencies
- Python 3.8+
- OpenCV
- Ultralytics YOLO (`pip install ultralytics`)
- NumPy
## Model Download
Object detection model link: https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

## How to Run
```bash
python main.py
