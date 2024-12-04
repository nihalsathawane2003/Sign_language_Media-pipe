from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time

# Initialize Flask app
app = Flask(__name__)

# Load models for different languages
models = {
    'american': pickle.load(open('american_model.p', 'rb'))['model'],
    'indian': pickle.load(open('indian_model.p', 'rb'))['model'],
    'german': pickle.load(open('german_model.p', 'rb'))['model']
}

# Set default language
current_language = 'american'

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Number of features the model expects (2 hands × 42 features each = 84 features)
expected_features = 84

# Global variables to store the predicted alphabet and current word
prediction_text = "No hands detected"
current_word = ""  # To store the word as it's formed
last_prediction_time = time.time()  # Track time of last prediction update
hand_removed_time = time.time()  # Track when the hand was removed

# Function to generate video stream
def generate_frames():
    global prediction_text, current_word, last_prediction_time, hand_removed_time  # Access the global variables

    cap = cv2.VideoCapture(0)
    stable_prediction = None  # To store the stable prediction
    stable_time = 1  # Time (in seconds) to wait for stability

    while True:
        data_aux = []
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = hands.process(frame_rgb)

        prediction_text = "No hands detected"  # Default text when no hands are detected

        if results.multi_hand_landmarks:
            # Sort hand landmarks by visibility (z-index) to maintain consistency
            hands_list = sorted(
                results.multi_hand_landmarks, key=lambda hand: hand.landmark[0].z
            )

            # Process up to 2 hands
            for hand_landmarks in hands_list[:2]:  # Take at most two hands
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            # If fewer than 2 hands detected, pad features for the missing hand
            if len(data_aux) < expected_features:
                data_aux.extend([0] * (expected_features - len(data_aux)))

            # Convert to NumPy array and reshape
            input_features = np.asarray(data_aux).reshape(1, -1)

            # Predict using the current language model
            prediction = models[current_language].predict(input_features)
            predicted_alphabet = prediction[0]  # Get predicted alphabet
            prediction_text = f"Prediction: {predicted_alphabet}"

            # Check for stability in hand movement
            current_time = time.time()
            if stable_prediction != predicted_alphabet:
                # If prediction changes, reset the timer
                stable_prediction = predicted_alphabet
                last_prediction_time = current_time
            else:
                # If prediction is stable for the defined time, update the word
                if current_time - last_prediction_time >= stable_time:
                    if predicted_alphabet != 'None' and (predicted_alphabet != current_word[-1] if current_word else True):
                        current_word += predicted_alphabet
                        last_prediction_time = current_time  # Reset the timer after adding the letter

            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # Overlay the prediction text on the frame
        cv2.putText(
            frame,
            prediction_text,
            (10, 50),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            (0, 255, 0),  # Text color (green)
            2,  # Thickness
            cv2.LINE_AA,
        )

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')  # HTML page for video feed

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def get_prediction():
    global prediction_text, current_word

    # Only return a response if the prediction has changed
    if prediction_text != get_prediction.last_prediction_text:
        get_prediction.last_prediction_text = prediction_text
        return jsonify({'prediction': prediction_text, 'current_word': current_word})
    else:
        # If no new prediction, return a simple response without the prediction text
        return jsonify({'prediction': None, 'current_word': current_word})

# Initialize last_prediction_text variable
get_prediction.last_prediction_text = "No hands detected"

@app.route('/reset_word', methods=['POST'])
def reset_word():
    # Reset the current word
    global current_word, last_prediction_time, hand_removed_time
    current_word = ""
    last_prediction_time = time.time()  # Reset the prediction time
    hand_removed_time = time.time()  # Reset the hand removal time
    return jsonify({'message': 'Word reset successful'})

@app.route('/set_language', methods=['POST'])
def set_language():
    global current_language
    # Get the selected language from the request
    language = request.json.get('language')
    
    if language in models:
        current_language = language
        return jsonify({'message': f'Language set to {language}'}), 200
    else:
        return jsonify({'error': 'Invalid language'}), 400

@app.route('/templates/<language>', methods=['GET'])
def select_language(language):
    """Select language and render the corresponding sign language template."""
    global current_language, models, prediction_text, current_word
    
    # Check if the requested language is in the available models
    if language in models:
        # Set the current language model
        current_language = language
        model = models[current_language]
    else:
        # If invalid language, return an error or fallback to a default language
        return jsonify({'error': 'Invalid language'}), 400

    # Pass current word and prediction to the template for rendering
    return render_template(f'{language}.html', 
                           language=current_language, 
                           detected_word=current_word,
                           prediction=prediction_text)
@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Handle image upload and prediction
    pass

@app.route('/help_american')
def help_american():
    return render_template('help_american.html')  # Template for American Sign Language help

@app.route('/help_indian')
def help_indian():
    return render_template('help_indian.html')  # Template for Indian Sign Language help

@app.route('/help_german')
def help_german():
    return render_template('help_german.html')  # Template for German Sign Language help


if __name__ == '__main__':
    app.run(debug=True)
