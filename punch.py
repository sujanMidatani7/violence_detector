# # import tensorflow as tf
# # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# # import tensorflow_hub as hub
# # import cv2
# # import numpy as np
# # import mediapipe as mp
# # import pandas as pd
# # import keras
# # import threading
# # print(keras.__version__)
# # # Initializations for MoveNet
# # gpus = tf.config.experimental.list_physical_devices('GPU')
# # for gpu in gpus:
# #     tf.config.experimental.set_memory_growth(gpu, True)

# # model_movenet = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
# # movenet = model_movenet.signatures['serving_default']

# # # Initializations for MediaPipe
# # mpPose = mp.solutions.pose
# # pose = mpPose.Pose()
# # mpDraw = mp.solutions.drawing_utils
# # model_mediapipe = keras.models.load_model("./lstm-model.h5")

# # lm_list = []
# # label="neutral"

# # # Your function definitions for the MoveNet model
# # # loop_through_people, draw_keypoints, etc...
# # def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
# #     for person in keypoints_with_scores:
# #         draw_connections(frame, person, edges, confidence_threshold)
# #         draw_keypoints(frame, person, confidence_threshold)
# # def draw_keypoints(frame, keypoints, confidence_threshold):
# #     y, x, c = frame.shape
# #     shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
# #     for kp in shaped:
# #         ky, kx, kp_conf = kp
# #         if kp_conf > confidence_threshold:
# #             cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)
# # def draw_connections(frame, keypoints, edges, confidence_threshold):
# #     y, x, c = frame.shape
# #     shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
# #     for edge, color in edges.items():
# #         p1, p2 = edge
# #         y1, x1, c1 = shaped[p1]
# #         y2, x2, c2 = shaped[p2]
        
# #         if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
# #             cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)
# # EDGES = {
# #     (0, 1): 'm',
# #     (0, 2): 'c',
# #     (1, 3): 'm',
# #     (2, 4): 'c',
# #     (0, 5): 'm',
# #     (0, 6): 'c',
# #     (5, 7): 'm',
# #     (7, 9): 'm',
# #     (6, 8): 'c',
# #     (8, 10): 'c',
# #     (5, 6): 'y',
# #     (5, 11): 'm',
# #     (6, 12): 'c',
# #     (11, 12): 'y',
# #     (11, 13): 'm',
# #     (13, 15): 'm',
# #     (12, 14): 'c',
# #     (14, 16): 'c'
# # }
# # # Your function definitions for the MediaPipe model
# # # make_landmark_timestep, draw_landmark_on_image, etc...
# # def detect(model, lm_list):
# #     global label
# #     lm_list = np.array(lm_list)
# #     lm_list = np.expand_dims(lm_list, axis=0)
# #     result = model.predict(lm_list)
# #     if result[0][0] > 0.5:
# #         label = "punch"
# #     else:
# #         label = "neutral"
# #     return str(label)
# # def draw_landmark_on_image(mpDraw, results, frame):
# #     mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
# #     for id, lm in enumerate(results.pose_landmarks.landmark):
# #         h, w, c = frame.shape
# #         print(id, lm)
# #         cx, cy = int(lm.x * w), int(lm.y * h)
# #         cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
# #     return frame
# # def make_landmark_timestep(results):
# #     print(results.pose_landmarks.landmark)
# #     c_lm = []
# #     for id, lm in enumerate(results.pose_landmarks.landmark):
# #         c_lm.append(lm.x)
# #         c_lm.append(lm.y)
# #         c_lm.append(lm.z)
# #         c_lm.append(lm.visibility)
# #     return c_lm
# # def draw_class_on_image(label, img):
# #     font = cv2.FONT_HERSHEY_SIMPLEX
# #     bottomLeftCornerOfText = (10,30)
# #     fontScale = 1
# #     if label == "punch":
# #         fontColor = (0,0,255)
# #     else:
# #         fontColor = (0,255,0)
# #     thickness = 2
# #     lineType = 2
# #     cv2.putText(img, str(label),
# #                 bottomLeftCornerOfText,
# #                 font,
# #                 fontScale,
# #                 fontColor,
# #                 thickness,
# #                 lineType)
# #     return img
# # def integrated_pose_detection(frame):
# #     # Process with MoveNet
# #     global lm_list
# #     global label
# #     img = frame.copy()
# #     img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)
# #     input_img = tf.cast(img, dtype=tf.int32)
# #     results_movenet = movenet(input_img)
# #     keypoints_with_scores = results_movenet['output_0'].numpy()[:,:,:51].reshape((6, 17, 3))
# #     loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)

# #     # Process with Mediapipe
# #     frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     results_mediapipe = pose.process(frameRGB)
    
# #     if results_mediapipe.pose_landmarks:
# #         lm = make_landmark_timestep(results_mediapipe)
# #         lm_list.append(lm)
# #         if len(lm_list) == 20:
# #             t1 = threading.Thread(target=detect, args=(model_mediapipe, lm_list,))
# #             t1.start()
# #             lm_list = []
# #             label="neutral"
        
# #         frame = draw_landmark_on_image(mpDraw, results_mediapipe, frame)
# #         frame = draw_class_on_image(label, frame)

# #     return frame

# # cap = cv2.VideoCapture(0)

# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if ret:
# #         frame = integrated_pose_detection(frame)
# #         cv2.imshow('Integrated Pose Detection', frame)
# #         if cv2.waitKey(10) & 0xFF == ord('q'):
# #             break

# # cap.release()
# # cv2.destroyAllWindows()
# import tensorflow as tf
# import tensorflow_hub as hub
# import cv2
# from matplotlib import pyplot as plt
# import numpy as np
# # Optional if you are using a GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
# movenet = model.signatures['serving_default']
# def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
#     for person in keypoints_with_scores:
#         draw_connections(frame, person, edges, confidence_threshold)
#         draw_keypoints(frame, person, confidence_threshold)
# def draw_keypoints(frame, keypoints, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
#     for kp in shaped:
#         ky, kx, kp_conf = kp
#         if kp_conf > confidence_threshold:
#             cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)
# EDGES = {
#     (0, 1): 'm',
#     (0, 2): 'c',
#     (1, 3): 'm',
#     (2, 4): 'c',
#     (0, 5): 'm',
#     (0, 6): 'c',
#     (5, 7): 'm',
#     (7, 9): 'm',
#     (6, 8): 'c',
#     (8, 10): 'c',
#     (5, 6): 'y',
#     (5, 11): 'm',
#     (6, 12): 'c',
#     (11, 12): 'y',
#     (11, 13): 'm',
#     (13, 15): 'm',
#     (12, 14): 'c',
#     (14, 16): 'c'
# }
# def draw_connections(frame, keypoints, edges, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
#     for edge, color in edges.items():
#         p1, p2 = edge
#         y1, x1, c1 = shaped[p1]
#         y2, x2, c2 = shaped[p2]
        
#         if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
#             cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
    
#     # Resize image
#     img = frame.copy()
#     img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192,256)
#     input_img = tf.cast(img, dtype=tf.int32)
    
#     # Detection section
#     results = movenet(input_img)
#     keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
#     # Render keypoints 
#     loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
    
#     cv2.imshow('Movenet Multipose', frame)
    
#     if cv2.waitKey(10) & 0xFF==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import keras
import threading
import os
# Initializations for MoveNet
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import tarfile

tar_path = r'C:\\Users\\hp\\OneDrive\\Desktop\\punch\\.venv\\movenet_multipose_lightning_1.tar.gz'
extract_path =r'C:\\Users\\hp\\OneDrive\\Desktop\\punch\\.venv\\movenet_multipose_lightning_1'

with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall(path=extract_path)

model_movenet = hub.load(r'C:\\Users\\hp\\OneDrive\\Desktop\\punch\\.venv\\movenet_multipose_lightning_1')
movenet = model_movenet.signatures['serving_default']

# Initializations for MediaPipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils 
model_mediapipe = keras.models.load_model("lstm-model.h5")

lm_list = []
label="neutral"

# Your function definitions for the MoveNet model
# loop_through_people, draw_keypoints, etc...
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
# Your function definitions for the MediaPipe model
# make_landmark_timestep, draw_landmark_on_image, etc...
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    if result[0][0] > 0.5:
        label = "punch"
    else:
        label = "neutral"
    return str(label)
def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = frame.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame
def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm
def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale = 1
    if label == "punch":
        fontColor = (0,0,255)
    else:
        fontColor = (0,255,0)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img
def integrated_pose_detection(frame):
    # Process with MoveNet
    global lm_list
    global label
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 256)
    input_img = tf.cast(img, dtype=tf.int32)
    results_movenet = movenet(input_img)
    keypoints_with_scores = results_movenet['output_0'].numpy()[:,:,:51].reshape((6, 17, 3))
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)

    # Process with Mediapipe
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_mediapipe = pose.process(frameRGB)
    
    if results_mediapipe.pose_landmarks:
        lm = make_landmark_timestep(results_mediapipe)
        lm_list.append(lm)
        if len(lm_list) == 20:
            t1 = threading.Thread(target=detect, args=(model_mediapipe, lm_list,))
            t1.start()
            lm_list = []
            label="neutral"
        
        frame = draw_landmark_on_image(mpDraw, results_mediapipe, frame)
        frame = draw_class_on_image(label, frame)

    return frame

while True:
    input_source = input("Enter 'w' to use the webcam or 'v' to use a video file: ").strip().lower()
    if input_source == 'w':
        cap = cv2.VideoCapture(0)
        break
    elif input_source == 'v':
        video_path = input("Enter the path of the video file: ").strip('\'\"')
        if not os.path.isfile(video_path):
            print("Invalid video file path. Please provide a valid path.")
        else:
            cap = cv2.VideoCapture(video_path)
            break
    else:
        print("Invalid input. Enter 'w' for webcam or 'v' for video file.")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = integrated_pose_detection(frame)
        cv2.imshow('Integrated Pose Detection', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
