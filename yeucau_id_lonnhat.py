import cv2
from ultralytics import YOLO
import os

# Đường dẫn file
path = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(path, "yolov8n.pt"))
cap = cv2.VideoCapture(os.path.join(path, "mvideo.mp4"))

max_id = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # QUAN TRỌNG: Phải dùng .track thay vì gọi trực tiếp model() để có ID
    results = model.track(frame, persist=True, classes=[2, 7], conf=0.3)
    
    # Vẽ các khung nhận diện lên frame
    annotated_frame = results[0].plot()

    # Kiểm tra xem có xe nào trong frame và có ID không
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Lấy danh sách tất cả ID đang có trong frame hiện tại
        ids = results[0].boxes.id.int().cpu().tolist()
        
        # Tìm ID lớn nhất trong danh sách vừa lấy
        current_max = max(ids)
        
        # Cập nhật ID lớn nhất từ trước đến nay
        if current_max > max_id:
            max_id = current_max

    # Hiển thị ID lớn nhất lên màn hình
    cv2.putText(annotated_frame, f"ID Max: {max_id}", (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

    cv2.imshow("Kiem tra ID lon nhat", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()