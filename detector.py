import cv2
import numpy as np

# vehicle reference classifiers
cars_classifier = cv2.CascadeClassifier('classifiers/cars.xml')

# read in video
video = cv2.VideoCapture('reference_vids/4K_traffic.webm')

while video.isOpened():

    # read in frame
    ret, frame = video.read()

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # pass frames to classifier
    cars = cars_classifier.detectMultiScale(gray, 1.4, 2)

    # extract bounding box for identified bodies
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('cars', frame)

    # break if q is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] cancelled, exiting...")
        break

video.release()
cv2.destroyAllWindows()