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
    count_large = 0

    for box in results[0].boxes:
        w, h = box.xywh[0][2:4] # Lấy chiều rộng và cao
        area = w * h
        if area > 2000:
            count_large += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"S:{int(area)}", (x1, y1-10), 0, 0.5, (0, 255, 0), 1)

    cv2.putText(annotated_frame, f"Xe lon (>2000): {count_large}", (30, 50), 2, 1, (0, 255, 0), 2)
    cv2.imshow("Loc kich thuoc", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()