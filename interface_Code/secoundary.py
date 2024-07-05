from cProfile import label
import cv2
import mediapipe as mp
import numpy as np
import keras
import threading

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = keras.models.load_model("lstm-model.h5")

people_info = []

def make_landmark_timestep(results):
    person_landmarks = []
    for person_idx, landmark in enumerate(results.pose_landmarks.landmark):
        c_lm = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        person_landmarks.append(c_lm)
    person_landmarks = np.array(person_landmarks)
    person_landmarks = np.expand_dims(person_landmarks, axis=0)
    person_landmarks = np.repeat(person_landmarks, 20, axis=0)  # Repeat for 20 time steps
    return person_landmarks



def draw_landmarks_on_image(mpDraw, results, frame):
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        h, w, c = frame.shape
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_classes_on_image(people_info, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    lineType = 2
    for person in people_info:
        label = person['label']
        x_coordinate = person['x_coordinate']
        y_coordinate = person['y_coordinate']
        bottomLeftCornerOfText = (x_coordinate[0], y_coordinate[0])
        
        # If neutral, draw a green rectangle and put a text label
        if label == "neutral":
            cv2.rectangle(img=img,
                          pt1=(min(x_coordinate), max(y_coordinate)),
                          pt2=(max(x_coordinate), min(y_coordinate) - 25),
                          color=(0, 255, 0),
                          thickness=1)
            cv2.putText(img, str(label),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        (0, 255, 0),  # Green color for neutral label
                        thickness,
                        lineType)
        # If punch, draw a red rectangle and put a text label
        elif label == "punch":
            cv2.rectangle(img=img,
                          pt1=(min(x_coordinate), max(y_coordinate)),
                          pt2=(max(x_coordinate), min(y_coordinate) - 25),
                          color=(0, 0, 255),
                          thickness=3)
            cv2.putText(img, str(label),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        (0, 0, 255),  # Red color for punch label
                        thickness,
                        lineType)
    return img


def detect_person(model, person_landmarks):
    person_landmarks = np.array(person_landmarks)
    person_landmarks = person_landmarks.reshape( 20, 132)  # Reshape to match expected input shape
    person_landmarks = np.expand_dims(person_landmarks, axis=0)
    result = model.predict(person_landmarks)
    if result[0][0] > 0.5:
        label = "punch"
    else:
        label = "neutral"
    return label


while True:
    choice = input("Enter 'c' to use the camera or 'v' to use a pre-existing video: ")
    if choice == 'c':
        cap = cv2.VideoCapture(0)  # Use webcam
        break
    elif choice == 'v':
        video_path = input("Enter the video file path: ").strip('"')
        cap = cv2.VideoCapture(video_path)  # Use pre-existing video
        break
    else:
        print("Invalid choice. Please enter 'c' or 'v'.")

i = 0
warm_up_frames = 60

# ...
while True:
    ret, frame = cap.read()

    # Check if the frame is empty or None
    if not ret:
        print("No frame captured or end of video.")
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    i = i + 1
    if i > warm_up_frames:
        print("Start detecting.")
        if results.pose_landmarks:
            person_landmarks = make_landmark_timestep(results)
            people_info = []
            label = detect_person(model, person_landmarks)
            x_coordinate = []
            y_coordinate = []
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                x_coordinate.append(cx)
                y_coordinate.append(cy)
            people_info.append({'label': label, 'x_coordinate': x_coordinate, 'y_coordinate': y_coordinate})

            frame = draw_landmarks_on_image(mpDraw, results, frame)
            frame = draw_classes_on_image(people_info, frame)

        cv2.imshow("image", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
