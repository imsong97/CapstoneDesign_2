from flask import Flask, render_template, Response
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf

app = Flask(__name__)

face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]

@app.route("/")
def index() :
    return render_template('index.html', data=EMOTIONS)

@app.route('/calc')
def calc() :  
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_frame() :
    cap = cv2.VideoCapture(0)
    emotion_classifier = load_model('emotion_model.hdf5')

    while True :
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert color to gray scale
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)) # Face detection in frame

        if len(faces) > 0:
            # For the largest image
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2) # Assign labeling
            
            # Resize the image to 48x48 for neural network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # # Emotion predict
            preds = emotion_classifier.predict(roi)
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')

if __name__ == "__main__" :
    app.run()
   
