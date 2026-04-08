import cv2
import numpy as np
from ultralytics import YOLO
import os
import easyocr

# 1. Khởi tạo đường dẫn
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'yolov8n.pt')
video_path = os.path.join(base_dir, 'plate2.mp4')

# Khởi tạo EasyOCR chạy trên CPU
print("Dang khoi tao bo doc chu (OCR)...")
reader = easyocr.Reader(['en'], gpu=False) 

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

READ_ZONE_Y = 450 
frame_skip = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_skip += 1
    # Vẽ vạch ranh giới
    cv2.line(frame, (0, READ_ZONE_Y), (frame.shape[1], READ_ZONE_Y), (0, 255, 255), 2)

    # YOLO quét xe
    results = model(frame, classes=[2, 3], conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Chỉ xử lý khi xe lại gần (vượt vạch vàng)
            if y2 >= READ_ZONE_Y:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Cắt vùng biển số
                roi_h = y2 - y1
                plate_roi = frame[y1 + int(roi_h*0.5):y2, x1:x2]
                
                if plate_roi.size > 0:
                    # Làm sạch điểm ảnh để trích xuất
                    zoom = cv2.resize(plate_roi, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
                    gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
                    clean = cv2.bilateralFilter(gray, 11, 75, 75)
                    
                    # Chỉ đọc chữ mỗi 10 khung hình để máy không bị giật (do dùng CPU)
                    if frame_skip % 10 == 0:
                        ocr_res = reader.readtext(clean)
                        result_text = ""
                        for (bbox, text, prob) in ocr_res:
                            if prob > 0.3:
                                result_text += text.upper() + " "
                        
                        if result_text:
                            # In ra Terminal để bạn theo dõi nếu cửa sổ bị lỗi
                            print(f">>> BIEN SO TRICH XUAT: {result_text}")
                            cv2.putText(frame, result_text, (x1, y1 - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Hiển thị vùng cận cảnh (Bọc trong try-except để tránh sập code)
                    try:
                        cv2.imshow("Goc Can", clean)
                    except:
                        pass

    # Hiển thị màn hình chính
    try:
        cv2.imshow("He thong Nhan dien & Trich xuat", frame)
    except Exception as e:
        print(f"Loi hien thi: {e}. Vui lang kiem tra lai buoc 1.")
        # Nếu vẫn lỗi hiển thị, ta in kết quả ra Terminal thay thế
        
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()