from keras.models import model_from_json
import cv2
import numpy as np

# Load model
json_file = open("signlanguagedetectionmodel128x128.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel128x128.h5")

# Function to preprocess the input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,128,128,1)
    return feature / 255.0

# Initialize webcam feed
cap = cv2.VideoCapture(0)

# Labels corresponding to the model's output classes
label =['A', 'B', 'C', 'blank']

while True:
    # Read frame from webcam
    _, frame = cap.read()

    # Draw rectangle for ROI
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    cropframe = frame[40:300, 0:300]
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (128, 128))

    # Extract features and make prediction
    cropframe = extract_features(cropframe)
    pred = model.predict(cropframe)
    prediction_label = label[pred.argmax()]

    # Display prediction result
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display output frame
    cv2.imshow("output", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
