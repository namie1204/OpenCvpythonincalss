import cv2
#Tăng giảm theo tỉ lệ
# Đọc ảnh
img = cv2.imread('anhcho1.png')

# Thay đổi kích thước ảnh
new_size = (699, 555)  # Đặt kích thước mới
resized_img = cv2.resize(img, new_size)

# Lưu ảnh đã thay đổi kích thước
cv2.imwrite('anh1.jpg', resized_img)
