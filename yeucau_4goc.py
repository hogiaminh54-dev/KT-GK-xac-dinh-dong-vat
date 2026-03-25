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
    annotated_frame = frame.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Vẽ 4 chấm tròn ở 4 góc
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for pt in corners:
            cv2.circle(annotated_frame, pt, 7, (0, 0, 255), -1)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv2.imshow("4 Goc", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()