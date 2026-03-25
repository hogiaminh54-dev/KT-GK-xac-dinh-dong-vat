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
    annotated_frame = results[0].plot(line_width=5) # line_width làm to khung và chữ label

    count = len(results[0].boxes)
    # Màu xanh dương (255, 0, 0) hoặc xanh lá (0, 255, 0). Ở đây dùng xanh dương cho nổi:
    cv2.putText(annotated_frame, f"Count: {count}", (30, 100), 
                cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Font va Mau", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()