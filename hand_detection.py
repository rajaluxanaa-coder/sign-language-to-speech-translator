import cv2
import mediapipe as mp

# New import syntax for MediaPipe 0.10.33
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize hand landmarker (CORRECT for video/image mode)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE  # Changed from VIDEO to IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# Start webcam
cap = cv2.VideoCapture(0)

print("✅ Webcam opened. Show your hand to the camera.")
print("Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands
    detection_result = detector.detect(mp_image)
    
    # Draw landmarks
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw dots (landmarks)
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            # Draw connections between landmarks
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                start = hand_landmarks[connection[0]]
                end = hand_landmarks[connection[1]]
                start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    cv2.imshow("Sign Language Translator - Hand Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Program closed")
