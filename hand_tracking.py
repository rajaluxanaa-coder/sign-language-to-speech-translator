import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Start webcam
cap = cv2.VideoCapture(0)

print("✅ Show your HAND to the camera")
print("Press 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Flip image (mirror effect)
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    result = hands.process(frame_rgb)
    
    # Draw landmarks if hand detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Show the video
    cv2.imshow("Sign Language Translator", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program closed")
