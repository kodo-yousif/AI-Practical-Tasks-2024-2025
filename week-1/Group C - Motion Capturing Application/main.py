import cv2
import time

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False
frame_count = 0
no_motion_start_time = None

# Sensitivity parameters
threshold_value = 50  # Threshold for motion detection
min_contour_area = 1000  # Minimum area to consider as motion
blur_kernel_size = (5, 5)  # Gaussian blur kernel size
frame_skip = 1  # Skip frames to improve performance
no_motion_timeout = 3  # Time in seconds after which recording stops if no motion
background_update_rate = 0.05  # Weight of the current frame for background update

print("Waiting for camera to stabilize...")
time.sleep(2)

ret, background_frame = cap.read()
if not ret:
    print("Error capturing initial background frame. Exiting...")
    cap.release()
    cv2.destroyAllWindows()

background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
background_frame = cv2.GaussianBlur(background_frame, blur_kernel_size, 0)

background_frame = background_frame.astype("float")

while cap.isOpened():
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, blur_kernel_size, 0)

    diff = cv2.absdiff(cv2.convertScaleAbs(background_frame), gray_frame)
    _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    cv2.accumulateWeighted(gray_frame, background_frame, background_update_rate)

    if motion_detected:
        if not recording:
            out = cv2.VideoWriter(f'motion_{int(time.time())}.avi', fourcc, 20.0, frame_size)
            recording = True
            print("Recording started...")
        no_motion_start_time = None
        out.write(frame1)

    if not motion_detected:
        if no_motion_start_time is None:
            no_motion_start_time = time.time()
        elif time.time() - no_motion_start_time > no_motion_timeout and recording:
            out.release()
            recording = False
            no_motion_start_time = None
            print("Recording stopped after 3 seconds of no motion...")

    cv2.imshow('Motion Detection', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

    frame_count += 1

cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
