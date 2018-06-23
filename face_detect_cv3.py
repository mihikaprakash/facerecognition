# Copyright Mihika Prakash

import cv3
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv3.CascadeClassifier(cascPath)

# Read the image
image = cv3.imread(imagePath)
gray = cv3.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv3.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv3.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv3.imshow("Faces found", image)
cv3.waitKey(0)

# Copyright Mihika Prakash
