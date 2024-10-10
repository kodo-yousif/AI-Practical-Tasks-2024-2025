# Title: 5- Head Movement
# Group E 

import cv2
import numpy as np
def smooth_transition(old_value, new_value, alpha=0.1):
    return old_value * (1 - alpha) + new_value * alpha
    # It uses an exponential moving average approach where alpha (0.1 here) controls how quickly the bounding box moves towards the new position.

# OpenCV has pre-trained models for face detection like the Haar Cascade classifier. For this, you can use haarcascade_frontalface_default.xml.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# VideoCapture is used to read video (record it )
cap = cv2.VideoCapture(0)

prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
# we always update this value to be the current positions

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # scaleFactor=1.05 is better for more accurate results but it reduces the processing
    # minSize is used for the size of the object that should be detected
    # it returns every detected face

    if len(faces) > 0:
        
        # Get the bounding box that contains all detected faces
        x_min = min([x for (x, y, w, h) in faces])
        y_min = min([y for (x, y, w, h) in faces])
        x_max = max([x + w for (x, y, w, h) in faces])
        y_max = max([y + h for (x, y, w, h) in faces])

        # Add padding
        padding = 50
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)

        # Smooth the transition for a smoother camera follow
        x_min = int(smooth_transition(prev_x, x_min))
        y_min = int(smooth_transition(prev_y, y_min))
        x_max = int(smooth_transition(prev_w, x_max))
        y_max = int(smooth_transition(prev_h, y_max))

        # Update previous values to be the current values
        prev_x, prev_y, prev_w, prev_h = x_min, y_min, x_max, y_max


        # Crop the image around the faces
        cropped_frame = frame[y_min:y_max, x_min:x_max]

        # Resize the cropped frame to match the original window size
        resized_frame = cv2.resize(cropped_frame, (frame.shape[1], frame.shape[0]))

        # Display the resulting frame
        cv2.imshow('Face Tracker', resized_frame)

    else:
        # If no faces are detected, show the full frame
        cv2.imshow('Face Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
