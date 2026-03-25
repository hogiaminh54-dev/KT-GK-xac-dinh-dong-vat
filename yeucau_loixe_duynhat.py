import cv2
from ultralytics import YOLO
import os

path = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(path, "yolov8n.pt"))
cap = cv2.VideoCapture(os.path.join(path, "mvideo.mp4"))

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Sửa ID ở đây: 2:car, 3:motorcycle, 5:bus, 7:truck
    results = model(frame, classes=[5]) # Ví dụ này chỉ đếm xe máy
    annotated_frame = results[0].plot()
    
    count = len(results[0].boxes)
    cv2.putText(annotated_frame, f"Chi dem xe bus: {count}", (30, 50), 2, 1, (0, 255, 255), 2)
    
    cv2.imshow("Loai xe duy nhat", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()