import cv2
import mediapipe as mp
import tensorflow as tf

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained model (you'll need to train this separately)
model = tf.keras.models.load_model('number_recognition_model.h5')

def preprocess_hand_image(image):
    # Implement preprocessing steps here
    # This could involve cropping, resizing, normalization, etc.
    return processed_image

def recognize_number(image):
    # Use the model to predict the number
    prediction = model.predict(image)
    return np.argmax(prediction)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand image
            hand_image = extract_hand_image(frame, hand_landmarks)

            # Preprocess the hand image
            processed_hand = preprocess_hand_image(hand_image)

            # Recognize the number
            number = recognize_number(processed_hand)

            # Display the recognized number
            cv2.putText(frame, f"Number: {number}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Number Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
