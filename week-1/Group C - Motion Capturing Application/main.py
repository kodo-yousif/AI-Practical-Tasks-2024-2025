import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False
no_motion_start_time = None

threshold_value = 20
min_contour_area = 500
blur_kernel_size = 5
no_motion_timeout = 3
background_update_rate = 0.5
sigmaX = 0

cap.read()

print("Waiting for camera to stabilize...")
time.sleep(1)

ret, background_frame = cap.read()
if not ret:
    print("Error capturing initial background frame. Exiting...")
    cap.release()
    cv2.destroyAllWindows()

background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
background_frame = cv2.GaussianBlur(background_frame, (blur_kernel_size, blur_kernel_size), 0)
background_frame = background_frame.astype("float")

def nothing(x):
    pass

cv2.namedWindow('Motion Detection')
cv2.createTrackbar('Threshold', 'Motion Detection', threshold_value, 255, nothing)
cv2.createTrackbar('Min Area', 'Motion Detection', min_contour_area, 5000, nothing)
cv2.createTrackbar('Blur Kernel', 'Motion Detection', blur_kernel_size, 20, nothing)
cv2.createTrackbar('No Motion Timeout', 'Motion Detection', no_motion_timeout, 10, nothing)
cv2.createTrackbar('BG Update Rate', 'Motion Detection', int(background_update_rate * 100), 100, nothing)
cv2.createTrackbar('SigmaX', 'Motion Detection', sigmaX, 1000, nothing)


while cap.isOpened():
    ret, consecutive_frame = cap.read()

    # cv2.imshow('Original', consecutive_frame)

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    threshold_value = cv2.getTrackbarPos('Threshold', 'Motion Detection')
    min_contour_area = cv2.getTrackbarPos('Min Area', 'Motion Detection')
    blur_kernel_size = cv2.getTrackbarPos('Blur Kernel', 'Motion Detection') * 2 + 1
    no_motion_timeout = cv2.getTrackbarPos('No Motion Timeout', 'Motion Detection')
    background_update_rate = cv2.getTrackbarPos('BG Update Rate', 'Motion Detection') / 100.0
    sigmaX = cv2.getTrackbarPos('SigmaX', 'Motion Detection')

    gray_frame = cv2.cvtColor(consecutive_frame, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('Grayed', gray_frame)

    gray_frame = cv2.GaussianBlur(gray_frame, (blur_kernel_size, blur_kernel_size), sigmaX)

    # cv2.imshow('Blurred', gray_frame)

    diff = cv2.absdiff(cv2.convertScaleAbs(background_frame), gray_frame)

    # cv2.imshow('Difference', diff)

    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # cv2.imshow('Threshold', thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(consecutive_frame, [contour], -1, (0, 255, 255), 2)
            motion_detected = True

    cv2.accumulateWeighted(gray_frame, background_frame, background_update_rate)

    if motion_detected:
        if not recording:
            out = cv2.VideoWriter(f'motion_{int(time.time())}.avi', fourcc, 20.0, frame_size)
            recording = True
            print("Recording started...")


        no_motion_start_time = None
        out.write(consecutive_frame)

    if not motion_detected:
        if no_motion_start_time is None:
            no_motion_start_time = time.time()
        elif time.time() - no_motion_start_time > no_motion_timeout and recording:
            out.release()
            recording = False
            no_motion_start_time = None
            print("Recording stopped after 3 seconds of no motion...")

    cv2.imshow('Motion Detection', consecutive_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
