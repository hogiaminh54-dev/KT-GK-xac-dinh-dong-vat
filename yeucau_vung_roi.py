import cv2
from ultralytics import YOLO
import os

path = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(path, "yolov8n.pt"))
cap = cv2.VideoCapture(os.path.join(path, "mvideo.mp4"))

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, classes=[2, 7])
    annotated_frame = frame.copy() # Dùng frame gốc để tự vẽ
    
    count_roi = 0
    # Vẽ vùng ROI (vùng bên trái màn hình)
    cv2.rectangle(annotated_frame, (0, 0), (320, 640), (0, 255, 255), 2)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) / 2
        # Nếu tâm xe nằm bên trái x < 320
        if center_x < 320:
            count_roi += 1
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(annotated_frame, f"Xe vung ROI: {count_roi}", (30, 50), 2, 1, (0, 255, 0), 2)
    cv2.imshow("Vung ROI", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
