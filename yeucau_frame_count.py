import cv2
from ultralytics import YOLO
import os

path = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(path, "yolov8n.pt"))
cap = cv2.VideoCapture(os.path.join(path, "mvideo.mp4"))
f_id = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    f_id += 1

    results = model(frame, classes=[2, 7])
    annotated_frame = results[0].plot()

    cv2.putText(annotated_frame, f"Frame: {f_id}", (30, 50), 2, 1, (255, 255, 255), 2)
    cv2.imshow("Dem Frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()