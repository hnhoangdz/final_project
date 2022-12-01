from flask import Flask, Response
import cv2
import cv2
import matplotlib.pyplot as plt
import dlib
import numpy as np
import argparse
import time
import torch
from models.finalv2 import Model
from data.dataset import DataTransform
from PIL import Image
import dlib
from imutils import face_utils
import imageio

# Initialize the objects
app = Flask(__name__)
video = cv2.VideoCapture(0)

# Face detection
detector = dlib.get_frontal_face_detector()

# Device
device = ("cuda: 0" if torch.cuda.is_available() else "cpu")

# Class labels
class_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

# Transform
transform = DataTransform(0, 255, 64)

# model
def load_model(weight_path, model):
    checkpoint = torch.load(weight_path, device)
    model.load_state_dict(checkpoint['params'], strict=False)
    model.eval()
    return model

weight_path = "/home/hoangdinhhuy/Hoang/project_fgw/final_project/checkpoints/finalv2_fer2013_64_AdamW_0.003/best.pth"
model = Model(1, 7)
model = load_model(weight_path, model)

@app.route('/')
def index():
    return "Hoang dep trai"

def process(video):
    # used to record the time when we processed last frame
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0
    while (True):
        predict_label = ""
        ret, frame = video.read()
        new_frame_time = time.time()
        frame = cv2.flip(frame, 1) 
        frame[:90, :220, :] = (45, 255, 255) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        results = detector(gray, 0)
        results = [face_utils.rect_to_bb(face) for face in results]
        
        if len(results) >= 1:
            for (x, y, w, h) in results:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x, y, w, h = results[0]
            face_bbox = gray[y:y+h, x:x+w]
            face_bbox = Image.fromarray(face_bbox)
            face_transformed = transform(face_bbox, phase="test")
            face_transformed = face_transformed.unsqueeze_(0)
            with torch.no_grad():
                outputs = model(face_transformed)
            predict_id = np.argmax(outputs.detach().numpy())
            predict_label = class_names[predict_id]
            
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        
        # converting the fps into integer
        fps = int(fps)
        
        # string fps
        fps = str(fps)
        
        cv2.putText(frame, "emotion: " + predict_label, (10,30), 1, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "fps: " + fps, (10,60), 1, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
        # frame = cv2.resize(frame, (640, 640))
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        
@app.route('/video_feed')
def video_feed():
    # Set to global because we refer the video variable on global scope,
    # Or in other words outside the function
    global video

    # Return the result on the web
    return Response(process(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)