import cv2
import os

xml_path = os.path.abspath('./haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(xml_path)

screen_width = 1280
screen_height = 720

stream = cv2.VideoCapture(0)

while (True):
    (grabbed, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        color = (0, 255, 255)
        stroke = 5
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    cv2.imshow("Face Reconition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
