# 🔥 Real-Time Fire & Object Detection with YOLOv8 and OpenCV

This project uses **YOLOv8** (from [Ultralytics](https://github.com/ultralytics/ultralytics)) and **OpenCV** to perform **real-time object detection** from a webcam feed and additionally detects **fire-like colors** (yellow/orange) in the video stream.

---

## 🚀 Features

* **YOLOv8 Object Detection**: Detects multiple classes of objects in real-time using the lightweight `yolov8n.pt` model.
* **Fire Color Detection**: Identifies regions in the video frame that contain fire-like colors (gold, yellow, orange, etc.) using **HSV color masking**.
* **Real-time Processing**: Works with a live camera feed (default webcam).
* **Annotated Output**: Displays detected objects and highlights possible fire regions with bounding boxes and labels.

---

## 📂 Project Structure

```
📦 fire-yolo-detection
 ┣ 📜 main.py          # Main script (the code provided)
 ┣ 📜 README.md        # Documentation
 ┗ 📜 requirements.txt # Python dependencies
```

---

## 🛠️ Requirements

* Python 3.8+
* [Ultralytics YOLO](https://docs.ultralytics.com)
* OpenCV
* NumPy

Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

---

## ▶️ Usage

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/fire-yolo-detection.git
   cd fire-yolo-detection
   ```
2. Download YOLOv8 weights (already included in the code as `yolov8n.pt`).
   You can replace with another YOLOv8 model (`yolov8s.pt`, `yolov8m.pt`, etc.) for better accuracy.
3. Run the script:

   ```bash
   python main.py
   ```
4. Press **`q`** to quit.

---

## 🔎 How It Works

1. **YOLOv8 Detection**

   * Loads `yolov8n.pt` and detects objects in each video frame.
   * Prints object labels & confidence scores to console.
   * Draws bounding boxes on detected objects.

2. **Fire Color Detection**

   * Converts each frame from BGR → HSV.
   * Applies **color masking** to detect warm colors associated with flames (yellow, orange, red, etc.).
   * Finds contours of the fire-like regions and highlights them with bounding boxes.
   * Displays `"🔥 FIRE COLOR"` above detected regions.

---

## 📸 Example Output

* Bounding boxes for YOLO objects.
* Highlighted regions in **orange boxes** for detected fire colors.
* Console log prints detections like:

  ```
  📦 PERSON (0.89)
  📦 BOTTLE (0.77)
  Fire Detected With Color!!!
  ```

---

## 📌 Notes

* The fire detection here is **color-based only** (not a deep learning fire detector).
* Works best under good lighting conditions.
* For production-grade fire detection, consider training YOLO with fire datasets.

---

## 📄 License

Creative Commons Zero v1.0 Universal License – free to use, modify, and distribute.

---
