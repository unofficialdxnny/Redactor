import cv2
import numpy as np

image_src = input("Image Name> ")
image_type = input("Type> ")

image = cv2.imread(f"{image_src}.{image_type}")

cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    face_roi = image[y:y + h, x:x + w]
    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
    image[y:y + h, x:x + w] = blurred_face

cv2.imwrite('blurred_image.jpg', image)
