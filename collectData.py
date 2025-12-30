import os
import cv2

# Directory setup
directory = 'SignImage128x128/'
print(os.getcwd())

if not os.path.exists(directory):
    os.mkdir(directory)
if not os.path.exists(f'{directory}/blank'):
    os.mkdir(f'{directory}/blank')

for i in range(65, 91):  # A-Z directories
    letter = chr(i)
    if not os.path.exists(f'{directory}/{letter}'):
        os.mkdir(f'{directory}/{letter}')

# Capture images
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    count = {
        'a': len(os.listdir(directory + "A")),
        'b': len(os.listdir(directory + "B")),
        'c': len(os.listdir(directory + "C")),
        'blank': len(os.listdir(directory + "blank"))
    }

    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    frame = frame[40:300, 0:300]  # Extract Region of Interest (ROI)
    cv2.imshow("ROI", frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (128, 128))  # Resize to 128x128

    # Save images based on key presses
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(os.path.join(directory + 'A/' + str(count['a']) + '.jpg'), frame)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(os.path.join(directory + 'B/' + str(count['b']) + '.jpg'), frame)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(os.path.join(directory + 'C/' + str(count['c']) + '.jpg'), frame)
    if interrupt & 0xFF == ord('.'):
        cv2.imwrite(os.path.join(directory + 'blank/' + str(count['blank']) + '.jpg'), frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
