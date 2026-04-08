import cv2
import imutils
import numpy as np
import os

# 1. Tự động xác định đường dẫn ảnh (Tránh lỗi 'NoneType')
# Lấy thư mục chứa file code hiện tại
base_dir = os.path.dirname(__file__)
image_path = os.path.join(base_dir, 'nhanDienBienSoXe.jpg')

# Đọc ảnh
image = cv2.imread(image_path)

# Kiểm tra nếu không tìm thấy file ảnh
if image is None:
    print("-" * 50)
    print(f"LỖI: Không tìm thấy ảnh tại: {image_path}")
    print("Giải pháp: Hãy đảm bảo file ảnh nằm cùng thư mục với file code này.")
    print("-" * 50)
    exit()

# 2. Tiền xử lý hình ảnh
image = imutils.resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh xám
gray = cv2.bilateralFilter(gray, 11, 17, 17)     # Lọc nhiễu nhưng giữ cạnh rõ
edged = cv2.Canny(gray, 30, 200)                 # Phát hiện các đường biên (cạnh)

# 3. Tìm các đường bao (Contours)
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# Sắp xếp các vùng tìm được theo diện tích to nhất
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

screenCnt = None
for c in cnts:
    # Tính chu vi và xấp xỉ hình dáng của vùng
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
    # Nếu hình dáng có đúng 4 góc -> Có khả năng cao là biển số
    if len(approx) == 4:
        screenCnt = approx
        break

# 4. Hiển thị kết quả nhận diện
if screenCnt is not None:
    # Vẽ khung màu xanh lá lên ảnh gốc để đánh dấu biển số
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    print("Kết quả: Đã xác định đúng vị trí biển số xe!")
    
    # Tạo mặt nạ để cắt (crop) riêng vùng biển số ra
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    
    # Hiện cửa sổ vùng biển số đã cắt
    cv2.imshow("Vung bien so (Cắt riêng)", Cropped)
else:
    print("Kết quả: Không tìm thấy vùng nào giống biển số xe.")

# Hiển thị ảnh tổng thể
cv2.imshow("Ket qua nhan dien", image)

print("Mẹo: Nhấn phím bất kỳ trên bàn phím để tắt các cửa sổ ảnh.")
cv2.waitKey(0)
cv2.destroyAllWindows()