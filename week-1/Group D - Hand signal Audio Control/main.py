import cv2
import mediapipe as mp
import pygame



mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

pygame.mixer.init()
pygame.mixer.music.load('music.mp3')

def classify_thumb_position(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
    wrist = landmarks[mp_hands.HandLandmark.WRIST].y

    if thumb_tip < thumb_mcp and thumb_tip < wrist:
        return "Thumbs Up"
    elif thumb_tip > thumb_mcp and thumb_tip > wrist:
        return "Thumbs Down"
    else:
        return "Unknown"

music_playing = False

try:
    with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame from webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    gesture = classify_thumb_position(landmarks)
                    if gesture == "Thumbs Up" and not music_playing:
                        pygame.mixer.music.play(-1)
                        music_playing = True
                    elif gesture == "Thumbs Down" and music_playing:
                        pygame.mixer.music.stop()
                        music_playing = False



except Exception as e:
    print("An error ")
finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
