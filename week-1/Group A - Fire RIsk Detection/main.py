import cv2
import threading
import betterplaysound
import numpy as np

# Load the fire detection cascade model
fire_cascade = cv2.CascadeClassifier('fire_detection_cascade_model.xml')

vid = cv2.VideoCapture(0)  # Use the default webcam

# Flags to control alarm status and fire detection
alarm_playing = False
fire_detected = False

# Adjusted function to check if the detected region has fire-like colors
def is_fire_color(frame, x, y, w, h):
    roi = frame[y:y + h, x:x + w]

    # Convert to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Expanded fire color range in HSV (covers more yellow, orange, red)
    lower_fire = np.array([5, 100, 100])  # Lower bound of fire-like hue
    upper_fire = np.array([35, 255, 255])  # Upper bound of fire-like hue

    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Check percentage of fire-colored pixels in the region
    fire_ratio = cv2.countNonZero(mask) / (w * h)
    print(f"Fire color ratio: {fire_ratio:.2f}")  # Log the fire color ratio

    # Lowered threshold for fire detection to avoid missing real fire
    return fire_ratio >= 0.08  # Now only 8% of the region needs to be fire-colored


# Function to play the fire alarm sound
def play_alarm_sound_function():
    global alarm_playing
    alarm_playing = True
    betterplaysound.playsound('fire_alarm.mp3', True)
    alarm_playing = False
    print("Fire alarm sound ended")


# Continuous loop to process video frames
while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to read frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5)

    if len(fire) > 0:
        for (x, y, w, h) in fire:
            # Only consider fire-like regions if they contain fire-like colors
            if is_fire_color(frame, x, y, w, h):
                cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 0, 0), 2)
                if not fire_detected:
                    print("Fire detected!")
                    fire_detected = True

                if not alarm_playing:
                    print("Starting fire alarm sound")
                    threading.Thread(target=play_alarm_sound_function).start()
            else:
                print("Detected non-fire light source, ignoring...")
    else:
        if fire_detected:
            print("Fire is no longer detected.")
            fire_detected = False

    cv2.imshow('Fire Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
