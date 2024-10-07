# Title: 5- Head Movement
# Group E 

import cv2
import numpy as np

# Initialize face detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# this detects the faces in the current frame

# Function for smoothing
def smooth_transition(old_value, new_value, alpha=0.1):
    return old_value * (1 - alpha) + new_value * alpha

# Open the video stream (0 for webcam)
cap = cv2.VideoCapture(0)

# Initialize variables for face tracking
prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Get the bounding box that contains all faces
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

        # Update previous values
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

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
