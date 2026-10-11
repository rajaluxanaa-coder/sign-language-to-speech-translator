import cv2
import mediapipe as mp
import pyttsx3

# Text to speech
engine = pyttsx3.init()

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            if landmarks[8].y < landmarks[6].y:
                text = "Hello"
            elif landmarks[12].y < landmarks[10].y:
                text = "Thank You"
            else:
                text = "Yes"

            cv2.putText(frame,text,(50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            engine.say(text)
            engine.runAndWait()

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()