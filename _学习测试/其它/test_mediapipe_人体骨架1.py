import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

handConStyle = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5)

cv2.namedWindow('MediaPipe Holistic', cv2.WINDOW_FULLSCREEN | cv2.WINDOW_KEEPRATIO)

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, img_src = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        img_src.flags.writeable = False
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

        results = holistic.process(img_src)

        # 画图
        img_dis = np.zeros((img_src.shape[0], img_src.shape[1], 3), np.uint8)

        # img_dis.flags.writeable = True
        # img_dis = cv2.cvtColor(img_dis, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            img_dis,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        mp_drawing.draw_landmarks(
            img_dis,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        mp_drawing.draw_landmarks(img_dis, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img_dis, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 右手21个节点坐标
        if results.right_hand_landmarks:
            for index, landmarks in enumerate(results.right_hand_landmarks.landmark):
                print(index, landmarks)
        # 鼻子坐标
        # print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE])

        cv2.imshow('MediaPipe Holistic', cv2.flip(cv2.resize(img_dis, None, None, 2, 2, cv2.INTER_NEAREST), 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

# https://blog.csdn.net/kalakalabala/article/details/121530651
