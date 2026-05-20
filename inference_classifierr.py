import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# 🔍 Auto camera detection
cap = None
for i in range(4):  # Try camera indexes 0, 1, 2, 3
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera opened with index {i}")
        break
    cap.release()

if not cap or not cap.isOpened():
    print(" Error: Camera not opening!")
    exit()

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label map (update this based on your training)
labels_dict = {
    0: 'selvi mam',
    1: 'jey sankar sir',
    2: 'santhosh',
    3: 'senthamil',
    4: 'thirumalai',
    5: 'E',
    6: 'F',
    7: 'G',
    8: 'H',
    9: 'I',
    10: 'J',
    11: '?'
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_label = int(prediction[0])
            print(" Predicted Label:", predicted_label)

            predicted_character = labels_dict.get(predicted_label, '?')

            # Draw result
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Show frame
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Exiting...")
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()
