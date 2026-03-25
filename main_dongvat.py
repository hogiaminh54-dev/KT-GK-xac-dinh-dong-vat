import cv2
from ultralytics import YOLO
import os

# 1. Thiết lập đường dẫn và chế độ Offline
path = os.path.dirname(os.path.abspath(__file__))
os.environ['ULTRALYTICS_OFFLINE'] = 'True' 

# 2. Load Model (Khuyên dùng yolov8n.pt để chạy mượt)
model_path = os.path.join(path, "yolov8n.pt") 
model = YOLO(model_path)

# 3. Mở Video động vật
video_path = os.path.join(path, "dongvat.mp4") 
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 4. Nhận diện tất cả động vật
    # Danh sách ID: 14:bird, 15:cat, 16:dog, 17:horse, 18:sheep, 19:cow...
    # Ở đây mình quét hết các loài phổ biến
    results = model(frame, conf=0.35, iou=0.45) 

    # 5. Dùng frame gốc để tự vẽ nhãn tùy chỉnh
    annotated_frame = frame.copy()

    # 6. Duyệt qua từng vật thể nhận diện được
    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Lấy tọa độ, độ tin cậy và ID loài
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            class_id = int(box.cls)
            label = model.names[class_id] # Lấy tên mặc định (dog, cat, bird...)

            # --- LOGIC GỘP: KIỂM TRA NẾU LÀ BIRD THÌ ĐỔI THÀNH GA ---
            color = (0, 255, 0) # Màu xanh mặc định
            if class_id == 14: # Nếu máy báo là 'bird'
                label = "Ga / Chim"
                color = (0, 0, 255) # Đổi khung sang màu đỏ cho nổi bật

            # Vẽ khung hình chữ nhật
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Vẽ nhãn chữ
            text = f"{label} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + 180, y1), color, -1) # Nền chữ
            cv2.putText(annotated_frame, text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 7. Đếm tổng số lượng hiển thị lên màn hình
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Tong so: {count}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("Nhan dien Dong Vat (Gop)", annotated_frame)

    # Nhấn 'q' để dừng
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()