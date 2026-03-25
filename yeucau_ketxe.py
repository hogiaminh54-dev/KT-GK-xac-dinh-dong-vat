import cv2
from ultralytics import YOLO
import os

path = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(path, "yolov8n.pt"))
cap = cv2.VideoCapture(os.path.join(path, "mvideo.mp4"))

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, classes=[2, 7], conf=0.4)
    annotated_frame = results[0].plot()
    car_count = len(results[0].boxes)

    # Logic kẹt xe
    if car_count > 10:
        cv2.putText(annotated_frame, "TRAFFIC JAM!", (150, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    
    cv2.putText(annotated_frame, f"Count: {car_count}", (30, 50), 2, 1, (0, 255, 0), 2)
    cv2.imshow("Ket Xe", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()