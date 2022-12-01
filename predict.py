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
# 
detector = dlib.get_frontal_face_detector()

# device
device = ("cuda: 0" if torch.cuda.is_available() else "cpu")

class_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

# transform
transform = DataTransform(0, 255, 64)

# model
def load_model(weight_path, model):
    checkpoint = torch.load(weight_path, device)
    model.load_state_dict(checkpoint['params'], strict=False)
    model.eval()
    return model

weight_path = "/home/hoangdinhhuy/Hoang/project_fgw/emotions_v2/checkpoints/finalv2_fer2013_64_AdamW_0.003/best.pth"
model = Model(1, 7)
model = load_model(weight_path, model)

# face detection
# detect_path = "/home/hoangdinhhuy/Hoang/project_fgw/emotions_v2/haar_cascades/haarcascade_frontalface_default.xml"
# detector = cv2.CascadeClassifier(detect_path)

# webcam
vid = cv2.VideoCapture(0)

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

image_list = []
while (True):
    predict_label = ""
    ret, frame = vid.read()
    new_frame_time = time.time()
    frame = cv2.flip(frame, 1) 
    frame[:90, :220, :] = (45, 255, 255) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # try:
    # results = detector.detectMultiScale(gray, scaleFactor=1.3,
    #                             minNeighbors=3, minSize=(30, 30),
    #                             flags=cv2.CASCADE_SCALE_IMAGE)
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
    x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_list.append(x)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

imageio.mimsave("video.gif", image_list, fps=30)