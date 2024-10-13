# Title : SmartLight Sensing Display
# Group : B
# Presenter : Ahmad Esmat Abdulwahid

import cv2
import screen_brightness_control as sbc


def get_average_light_intensity(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate the average light intensity (mean pixel value)
    avg_intensity = gray.mean()
    return avg_intensity


def adjust_brightness_based_on_light(avg_intensity):
    # Normalize the intensity to a range suitable for brightness (0 to 100)
    brightness = min(max(int(avg_intensity / 255 * 100), 10), 100)  # At least 10% brightness
    sbc.set_brightness(brightness)



# Initialize the camera
cap = cv2.VideoCapture(0)


try:
    while True:
        # act for knowing if the camera is working and ready
        # while frame will give us 1 frame of now to process it
        act, frame = cap.read()
        if not act:
            break

        # Get the average light intensity from the frame
        avg_intensity = get_average_light_intensity(frame)
        
        # Adjust the brightness based on the light intensity
        adjust_brightness_based_on_light(avg_intensity)

        # Show the frame for visual feedback (optional)
        cv2.imshow("Camera Feed", frame)

        # Press 'esc' to exit the loop 27 is the unicode of escape button
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # or press the close button
        if cv2.getWindowProperty("Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()



