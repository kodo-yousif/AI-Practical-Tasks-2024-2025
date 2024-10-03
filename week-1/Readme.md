# 1- Fire Risk Detection

Create a program that utilizes a computer's camera to identify fire risk, then produce a vocal alarm using an audio file.

## Requirements:

- Utilize the computer's camera to capture real-time video.
- The program should display a video feed, from a live camera feed.
- Process the video frames to identify the existence of fire.
- The program should announce the state audibly.
- Students can use publicly available models for fire detection

## Tools/Libraries Suggested:

- Python for coding
- Time library for adding debounce ( Extra )
- OpenCV for image processing
- Numpy ( If Needed )
- openCV models

# 2- SmartLight Sensing Display

Develop an application that adjusts computers brightness display based on the ambient light in its environment. and user can change the app sensevity while the program is running, In a dark room, the computers screen brightness is low. In a brightly room the screen should turn brightest.

## Requirements:

- Use the computer's camera to sense the ambient light.
- The program doesn't show a camera video.
- Dynamically adjust the computers display based on the light intensity detected.
- Ensure a smooth transition and accurate light detection for optimal display adjustment.

## Tools/Libraries Suggested:

- Python for coding
- Numpy ( If Needed )
- OpenCV for image processing

<b><i>Note: Should not use libraries to perform main goals</i></b>

# 3- Motion Capturing Application

Design a program that uses the computer's camera to capture the views when there is motion by saving the clip videos, and highlight the areas where the movements happened.

## Requirements:

- Continuously capture video from the computer's camera.
- The program should not display a video feed, from a live camera feed only.
- Process the video frames to detect and indicate motion.
- Provide clear and immediate motion detection results.

## Tools/Libraries Suggested:

- Python for coding
- Numpy ( If Needed )
- OpenCV for image processing

<b><i>Note: Should not use libraries to perform main goals</i></b>

# 4- Hand signal Audio Control

Create a program that uses computers camera to detect hand gestures and use the to control playing audio file to play/stop it.

## Requirements:

- Capture video from the computer’s camera. don't show the video
- Detect and track hand movement in real-time.
- Classify the hand signal
- ability to play the audio
- ability to stop the audio
- ability to increase the volume
- ability to decrees the volume

## Tools/Libraries Suggested:

- Python for coding
- OpenCV for image processing
- Numpy (if needed)
- PyCaw or another library to control system volume
- public OpenCv models to detect hand gesture or any other model

<b><i>Note: Should not use libraries to perform main goals</i></b>

# 5- Head Movement

Create a program that follows the face position in a video and crop the head from the video with some paddings, if more than one face is detected then fit all then in the cropped image, and only show the cropped video.

the goal is to create a face following program that follows the face in a smooth way

## Requirements:

- detect the faces.
- fit all the faces in the screen.
- follow the faces movement smoothly

## Tools/Libraries Suggested:

- Python for coding
- Numpy ( If Needed )
- OpenCV for image processing
- OpenCv face detection models

<b><i>Note: Should not use libraries to perform main goals</i></b>

# 6- Hand Gesture-Based Mouse Control

Create a program that uses hand gestures captured from a live camera feed to control the mouse on the screen. Movements of the hand should move the mouse when the hand gesture is classified as movement mode, and other gestures (like a pinch) should simulate mouse clicks.

## Requirements:

- Use the computer's camera to detect and track the hand in real-time. while no vide is showing.
- Move the mouse pointer based on the hand's position in the video feed when it is classified as movement gesture
- The mouse controls are ( movement, click, double click, right click, scroll)

## Tools/Libraries Suggested:

- Python for coding
- OpenCV for video capture and hand detection
- Mediapipe for hand tracking (or custom hand detection model using OpenCV)
- PyAutoGUI for controlling the mouse

<b><i>Note: Should not use libraries to perform main goals</i></b>
