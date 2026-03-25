import cv2
from ultralytics import YOLO
import os

path = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(path, "yolov8n.pt"))
cap = cv2.VideoCapture(os.path.join(path, "mvideo.mp4"))

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.track(frame, persist=True)
    # labels=False sẽ ẩn tên class và ID
    annotated_frame = results[0].plot(labels=False) 

    cv2.imshow("An ID", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()