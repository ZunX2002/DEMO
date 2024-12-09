import cv2
from flask import Flask, render_template, request, jsonify, Response
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
model = load_model('emotion_recognition_model.h5')

# Nhãn cảm xúc
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Hàm xử lý ảnh trước khi đưa vào mô hình
def preprocess_image(img):
    img = img.resize((48, 48)).convert('L')  # Resize và chuyển thành ảnh xám
    img = np.array(img) / 255.0  # Chuẩn hóa
    img = img.reshape(1, 48, 48, 1)  # Thêm batch dimension
    return img

# Hàm nhận diện cảm xúc từ webcam
def gen_frames():
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Phát hiện tất cả các khuôn mặt
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_normalized = face_gray / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            face_input = np.expand_dims(face_input, axis=-1)
            
            # Dự đoán cảm xúc cho mỗi khuôn mặt
            prediction = model.predict(face_input)
            emotion = emotion_labels[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # Vẽ hình chữ nhật quanh khuôn mặt và thêm nhãn cảm xúc
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({confidence*100:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        img = Image.open(file.stream)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)
        return jsonify({'emotion': emotion, 'confidence': float(confidence)})
    return jsonify({'error': 'No image uploaded'}), 400

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
