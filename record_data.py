import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Create folder to save data
data_dir = "sign_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define the signs we want to record
signs = ["Hello", "Yes", "No"]
current_sign_index = 0
current_sample = 0
samples_per_sign = 50  # Number of recordings per sign
recording = False

# To store landmarks
landmarks_sequence = []
all_data = []

print("=" * 50)
print("SIGN LANGUAGE DATA RECORDER")
print("=" * 50)
print(f"Signs to record: {signs}")
print(f"Samples per sign: {samples_per_sign}")
print("\nINSTRUCTIONS:")
print("1. Press SPACE to start recording for current sign")
print("2. Make the sign and hold it steady")
print("3. Recording stops automatically after 30 frames")
print("4. Press 'n' to go to NEXT sign")
print("5. Press 'q' to quit and save\n")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    # Draw hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks (21 points x 3 coordinates)
            if recording and len(landmarks_sequence) < 30:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_sequence.append(landmarks)
    
    # Display status
    current_sign = signs[current_sign_index]
    cv2.putText(frame, f"Current Sign: {current_sign}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Progress: {current_sample}/{samples_per_sign}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if recording:
        progress = len(landmarks_sequence)
        cv2.putText(frame, f"RECORDING: {progress}/30 frames", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, "SPACE: Record | n: Next Sign | q: Quit", (10, frame.shape[0]-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Sign Language Data Recorder", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # Space bar - start recording
        if not recording:
            recording = True
            landmarks_sequence = []
            print(f"Recording: {current_sign} - Sample {current_sample + 1}/{samples_per_sign}")
    
    elif key == ord('n'):  # Next sign
        if current_sample >= samples_per_sign:
            print(f"✓ Completed {current_sign}! Moving to next sign...")
            current_sign_index += 1
            current_sample = 0
            if current_sign_index >= len(signs):
                print("\n✅ ALL SIGNS RECORDED! Saving data...")
                break
        else:
            print(f"⚠️ Need {samples_per_sign - current_sample} more samples for {current_sign}")
    
    elif key == ord('q'):  # Quit
        print("\n⚠️ Exiting without saving...")
        break
    
    # When 30 frames are recorded, save the sequence
    if recording and len(landmarks_sequence) >= 30:
        # Save only first 30 frames
        landmarks_sequence = landmarks_sequence[:30]
        all_data.append({
            'sign': current_sign,
            'landmarks': landmarks_sequence
        })
        current_sample += 1
        recording = False
        print(f"✓ Saved sample {current_sample}/{samples_per_sign} for {current_sign}")
        
        # If we completed all samples for current sign
        if current_sample >= samples_per_sign:
            if current_sign_index + 1 < len(signs):
                print(f"\n✅ Completed {current_sign}! Press 'n' to go to next sign\n")
            else:
                print(f"\n✅ Completed ALL signs! Press 'n' to save and exit\n")

# Save all recorded data
if all_data:
    with open(os.path.join(data_dir, 'sign_language_data.pkl'), 'wb') as f:
        pickle.dump(all_data, f)
    print(f"\n✅ Data saved successfully!")
    print(f"   Total samples: {len(all_data)}")
    print(f"   Signs: {set([d['sign'] for d in all_data])}")
    print(f"   Location: {data_dir}/sign_language_data.pkl")
else:
    print("\n⚠️ No data recorded.")

cap.release()
cv2.destroyAllWindows()
