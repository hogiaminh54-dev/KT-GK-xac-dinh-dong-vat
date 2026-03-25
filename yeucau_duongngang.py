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
    annotated_frame = results[0].plot()

    # Vẽ đường ngang giữa màn hình
    h, w, _ = frame.shape
    cv2.line(annotated_frame, (0, h // 2), (w, h // 2), (255, 0, 0), 3)

    cv2.imshow("Duong Ngang", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()