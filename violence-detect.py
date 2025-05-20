import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.nn import softmax

# Load the model
model = load_model('violence_detection.h5')

# Define the video path
video_path = 'test-v.mp4'

# Define a function to preprocess video frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = preprocess_input(frame)
    return frame

# Open the video file or capture device
cap = cv2.VideoCapture(video_path)

# Initialize a list to store frames
frames = []

# Frame sequence length expected by the model
sequence_length = 16

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame and add it to the sequence
    preprocessed_frame = preprocess_frame(frame)
    frames.append(preprocessed_frame)

    # Once we have enough frames for a sequence, predict
    if len(frames) == sequence_length:
        # Convert list of frames to numpy array
        frame_sequence = np.array(frames)

        # Add batch dimension and predict
        frame_sequence = np.expand_dims(frame_sequence, axis=0)
        predictions = model.predict(frame_sequence)

        # Normalize the predictions
        normalized_predictions = softmax(predictions).numpy()

        # Reset frames list for next sequence
        frames = []

        # Display the normalized predictions
        print(f"Normalized Predictions: {normalized_predictions}")
    
    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
