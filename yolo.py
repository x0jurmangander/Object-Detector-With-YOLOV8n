import cv2
import numpy as np
from ultralytics import YOLO

# Init YOLO General Module
model = YOLO("/yolov8n.pt")

# Camera Init
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Detection with Fire Color", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed To Capture The Video!")
        break

    # ============ 🔶 1. Detect Objects Using YOLO ============
    results = model.predict(frame, stream=False, verbose=False)
    result = results[0]
    annotated_frame = result.plot()

    # Print Detected Objects
    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[class_id]
        print(f"📦 {label.upper()} ({conf:.2f})")

    # ============ 🔥 2. Detect Frames With Colors (Gold, Yellow) ============
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color Of Frame (Gold, Yellow) With HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    lower_warm_red = np.array([10, 100, 100])
    upper_warm_red = np.array([20, 255, 255])
    lower_benzene_flame = np.array([90, 50, 200])
    upper_benzene_flame = np.array([130, 150, 255])
    lower_warm_blue = np.array([10, 100, 100])
    upper_warm_blue = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Finding Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Ignore Tight Frames
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(annotated_frame, "🔥 FIRE COLOR", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            print("Fire Detected With Color!!!")

    # ============ Show Result ============
    cv2.imshow("Detection with Fire Color", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
