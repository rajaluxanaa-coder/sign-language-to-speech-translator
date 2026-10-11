import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the recorded data
data_path = "sign_data/sign_language_data.pkl"
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print("=" * 50)
print("TRAINING SIGN LANGUAGE MODEL")
print("=" * 50)
print(f"Loaded {len(data)} samples")

# Prepare data
X = []  # Landmarks
y = []  # Labels (Hello, Yes, No)

# Map sign names to numbers
sign_to_id = {"Hello": 0, "Yes": 1, "No": 2}
id_to_sign = {0: "Hello", 1: "Yes", 2: "No"}

for sample in data:
    landmarks = np.array(sample['landmarks'])  # Shape: (30, 63) - 30 frames x 21 points x 3
    X.append(landmarks)
    y.append(sign_to_id[sample['sign']])

X = np.array(X)
y = np.array(y)

print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")

# Convert labels to categorical (one-hot encoding)
y_categorical = to_categorical(y, num_classes=3)

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 63)),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 signs: Hello, Yes, No
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✅ Model Architecture:")
model.summary()

# Train the model
print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1
)

# Save the trained model
model.save("sign_language_model.keras")
print("\n✅ Model saved as 'sign_language_model.keras'")

# Print final accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n📊 Final Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the label mapping
import json
with open("sign_labels.json", "w") as f:
    json.dump(id_to_sign, f)
print("✅ Labels saved as 'sign_labels.json'")

# Plot training history (optional - requires matplotlib)
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("✅ Training chart saved as 'training_history.png'")
except:
    print("⚠️ matplotlib not installed - skipping chart")

print("\n🎉 TRAINING COMPLETE!")
print("   Ready for real-time translation!")
