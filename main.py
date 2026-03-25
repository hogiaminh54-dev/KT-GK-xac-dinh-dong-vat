import cv2
from ultralytics import YOLO
import os

# Ngắt kết nối mạng của YOLO để tránh lỗi SSL
os.environ['ULTRALYTICS_OFFLINE'] = 'True'

import cv2
from ultralytics import YOLO
import os

# Lấy đường dẫn chính xác của thư mục hiện tại
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, "yolov8n.pt")

# Load model bằng đường dẫn tuyệt đối
model = YOLO(model_path)

# Tương tự cho video
video_path = os.path.join(path, "mvideo.mp4")
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Đã hết video hoặc không thể đọc file.")
        break

    # 3. Nhận diện xe ô tô (class 2) và xe tải (class 7)
    # conf=0.4: chỉ lấy những xe có độ tin cậy trên 40%
    results = model(frame, classes=[2, 7], conf=0.4)

    # 4. Vẽ khung hình và lấy số lượng xe
    annotated_frame = results[0].plot() 
    car_count = len(results[0].boxes) 

    # Viết số lượng xe lên màn hình
    cv2.putText(annotated_frame, f"So luong xe: {car_count}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 5. Hiển thị cửa sổ kết quả
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Nhấn phím 'q' trên bàn phím để thoát sớm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()