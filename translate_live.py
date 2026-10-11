import cv2
import mediapipe as mp
import numpy as np
import json
import pyttsx3
from tensorflow.keras.models import load_model

# Load trained model and labels
model = load_model("sign_language_model.keras")
with open("sign_labels.json", "r") as f:
    id_to_sign = json.load(f)

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Store landmark sequence
landmarks_sequence = []
PREDICTION_THRESHOLD = 0.8  # Confidence threshold
last_prediction = ""
prediction_counter = 0

cap = cv2.VideoCapture(0)

print("=" * 50)
print("REAL-TIME SIGN LANGUAGE TO SPEECH TRANSLATOR")
print("=" * 50)
print("Make these signs:")
print("  👋 Hello  |  👍 Yes  |  ✊ No")
print("\nPress 's' to speak the detected sign")
print("Press 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    current_prediction = ""
    confidence = 0
    
    # Extract landmarks if hand detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract 63 landmarks (21 points x 3 coordinates)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Add to sequence (keep last 30 frames)
            landmarks_sequence.append(landmarks)
            if len(landmarks_sequence) > 30:
                landmarks_sequence.pop(0)
            
            # Predict if we have 30 frames
            if len(landmarks_sequence) == 30:
                input_data = np.array(landmarks_sequence).reshape(1, 30, 63)
                prediction = model.predict(input_data, verbose=0)
                confidence = np.max(prediction)
                predicted_id = np.argmax(prediction)
                current_prediction = id_to_sign[str(predicted_id)]
    
    # Display prediction on screen
    if current_prediction and confidence > PREDICTION_THRESHOLD:
        cv2.putText(frame, f"Sign: {current_prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Speak when 's' is pressed
        if prediction_counter > 0:
            cv2.putText(frame, f"SPEAKING: {current_prediction}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No sign detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Press 's' to speak | 'q' to quit", (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Sign Language to Speech Translator", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):  # Speak the detected sign
        if current_prediction and confidence > PREDICTION_THRESHOLD:
            engine.say(current_prediction)
            engine.runAndWait()
            prediction_counter = 10  # Show "SPEAKING" for 10 frames
        else:
            print("No sign detected. Please show a clear sign.")
    
    elif key == ord('q'):
        break
    
    if prediction_counter > 0:
        prediction_counter -= 1

cap.release()
cv2.destroyAllWindows()
print("👋 Translator closed")
