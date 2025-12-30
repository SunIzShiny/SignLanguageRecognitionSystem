from flask import Flask,render_template,Response
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

json_file = open("signlanguagedetectionmodel128x128.json")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel128x128.h5")

labels =['A', 'B', 'C', 'blank']

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 128, 128, 1)
    return feature / 255.0

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        if not _:
            break
        cropframe = frame[40:300, 0:300]
        cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe, (128, 128))


        cropframe = extract_features(cropframe)
        pred = model.predict(cropframe)
        prediction_label = labels[pred.argmax()]

        cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
        if prediction_label == 'blank':
            cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2, cv2.LINE_AA)
        else:
            accu = "{:.2f}".format(np.max(pred) * 100)
            cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# Define the route to stream video to browser
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
