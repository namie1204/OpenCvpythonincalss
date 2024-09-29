import cv2
import serial
import numpy as np
import urllib.request
from datetime import datetime
import os
import threading

# Tạo thư mục để lưu video và ảnh chụp
os.makedirs("ghi_hinh", exist_ok=True)

# Khởi tạo Serial để kết nối với Arduino
ser = serial.Serial('COM5', 250000)
#url = 'http://192.168.176.241/cam-hi.jpg'
url = 'http://192.168.100.160/cam-hi.jpg'
# Khởi tạo các biến toàn cục
is_recording = False
is_auto = True
out = None
frame_w, frame_h = 640, 480

# Khởi tạo face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'D:\OpenCv_VsCode\CV2trenlop\trainer\trainer.yml')
names = ['Nam','Son','An']  # Danh sách tên tương ứng với ID trong trainer

# Khởi tạo cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Khởi tạo mạng nơ-ron để phát hiện người (mobilenet SSD)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Tọa độ và kích thước các nút điều khiển
auto_button_width = 100
auto_button_height = 30
distance_from_auto = 150
BUTTONS = {
    "up": (frame_w - 150, frame_h + 10, frame_w - 80, frame_h + 50),
    "down": (frame_w - 150, frame_h + 60, frame_w - 80, frame_h + 100),
    "left": (frame_w - 230, frame_h + 30, frame_w - 170, frame_h + 70),
    "right": (frame_w - 50, frame_h + 30, frame_w + 10, frame_h + 70),
}

# Các hàm xử lý sự kiện ghi hình, chụp ảnh và sự kiện chuột
def start_recording():
    global is_recording, out
    if not is_recording:
        is_recording = True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = f"ghi_hinh/video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_w, frame_h))
        print(f"Bắt đầu ghi hình: {filename}")

def stop_recording():
    global is_recording, out
    if is_recording:
        is_recording = False
        if out:
            out.release()
        print("Dừng ghi hình.")

def capture_image(frame):
    filename = f"ghi_hinh/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Chụp ảnh: {filename}")

def mouse_callback(event, x, y, flags, param):
    global is_auto
    if event == cv2.EVENT_LBUTTONDOWN:
        if 8 <= x <= 120 and frame_h + 40 <= y <= frame_h + 70:
            capture_image(param)
        elif 140 <= x <= 240 and frame_h + 40 <= y <= frame_h + 70:
            if not is_recording:
                start_recording()
            else:
                stop_recording()
        elif 270 <= x <= 370 and frame_h + 40 <= y <= frame_h + 70:  # Nút "Auto"
            is_auto = not is_auto
            print("Chế độ Auto:", "Bật" if is_auto else "Tắt")
        
        if not is_auto:
            for btn, (x1, y1, x2, y2) in BUTTONS.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if btn == "up":
                        ser.write("u!".encode())
                        print("Gửi lệnh lên: u!")
                    elif btn == "down":
                        ser.write("d!".encode())
                        print("Gửi lệnh xuống: d!")
                    elif btn == "left":
                        ser.write("l!".encode())
                        print("Gửi lệnh trái: l!")
                    elif btn == "right":
                        ser.write("r!".encode())
                        print("Gửi lệnh phải: r!")


def fetch_frame():
    try:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        return cv2.imdecode(imgNp, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error fetching frame: {e}")
        return None

def face_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))

def video_stream_worker():
    global out, is_recording, is_auto
    while True:
        frame = fetch_frame()
        if frame is None or frame.size == 0:
            continue
        frame = cv2.flip(frame, 1)
        
        # Chế độ Auto: Sử dụng phát hiện khuôn mặt và điều khiển động cơ servo tự động
        if is_auto:
            faces = face_detection(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Nhận diện khuôn mặt và hiển thị tên và độ tin cậy
                id, confidence = recognizer.predict(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY))
                if confidence < 100 and id < len(names):
                    id = names[id]
                    confidence_text = "  {0}%".format(round(100 - confidence))
                else:
                    id = "nguoi la"
                    confidence_text = "  {0}%".format(round(100 - confidence))

                cv2.putText(frame, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, str(confidence_text), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                
            if len(faces) > 0:
                face_center_x = faces[0][0] + faces[0][2] / 2
                face_center_y = faces[0][1] + faces[0][3] / 2
                err_x = 30 * (face_center_x - frame_w / 2) / (frame_w / 2)
                err_y = 30 * (face_center_y - frame_h / 2) / (frame_h / 2)
                ser.write(f"{err_x:.2f}x!".encode())
                ser.write(f"{err_y:.2f}y!".encode())
                print(f"X: {err_x}, Y: {err_y}")
            else:
                ser.write("o!".encode())
        # Chế độ thủ công: Sử dụng MobileNet SSD để phát hiện người
        else:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    # Lấy chỉ số của class
                    idx = int(detections[0, 0, i, 1])

                    # Chỉ xử lý nếu đối tượng phát hiện là "person"
                    if CLASSES[idx] == "person":
                        # Tính toán tọa độ hộp giới hạn (bounding box)
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Vẽ hộp giới hạn xung quanh đối tượng
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                        # Chuyển đổi phần hình ảnh trong khung đối tượng sang thang độ xám để nhận diện khuôn mặt
                        gray_frame = cv2.cvtColor(frame[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY)

                        # Nhận diện khuôn mặt và hiển thị tên và độ tin cậy
                        # Nhận diện khuôn mặt và hiển thị tên và độ tin cậy
                        id, confidence = recognizer.predict(gray_frame)
                        if confidence < 100 and id < len(names):
                            id = names[id]
                            confidence_text = "  {0}%".format(round(100 - confidence))
                        else:
                            id = "Nguoi la"
                            confidence_text = "  {0}%".format(round(100 - confidence))


                        # Hiển thị thông tin nhận diện trên khung hình
                        cv2.putText(frame, str(id), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(frame, str(confidence_text), (startX, endY + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

                        # Hiển thị thông tin đối tượng đã phát hiện
                        label = f"Person: {confidence:.2f}"
                        cv2.putText(frame, label, (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    
            # Vẽ các nút điều khiển servo khi tắt chế độ auto
            # Vẽ các nút điều khiển servo khi tắt chế độ auto
            for btn, (x1, y1, x2, y2) in BUTTONS.items():
                color = (0, 200, 0) if btn == "up" else (200, 0, 0) if btn == "down" else (0, 0, 200) if btn == "left" else (200, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.putText(frame, btn.capitalize(), (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 

        # Hiển thị thời gian hiện tại và tọa độ
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, current_time, (frame_w - 100, frame_h - 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (8, frame_h + 40), (120, frame_h + 70), (0, 0, 255), -1)
        cv2.putText(frame, 'Chup anh', (18, frame_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        button_color = (255, 0, 0)
        button_text = 'Ghi hinh' if not is_recording else 'Dung ghi'
        cv2.rectangle(frame, (140, frame_h + 40), (240, frame_h + 70), button_color, -1)
        cv2.putText(frame, button_text, (150, frame_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (270, frame_h + 40), (370, frame_h + 70), (0, 255, 255), -1)
        cv2.putText(frame, 'Auto', (280, frame_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('Camera', frame)
        
        if is_recording and out:
            out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        cv2.setMouseCallback('Camera', mouse_callback, param=frame)

    if out:
        out.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_thread = threading.Thread(target=video_stream_worker)
    video_thread.start()
    video_thread.join()
