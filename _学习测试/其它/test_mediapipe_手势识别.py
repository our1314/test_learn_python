import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands  # 手部追踪模型
hands = mpHands.Hands(model_complexity=1, max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=5)
pTime = 0
cTime = 0

last_x = 0
last_y = 0
while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        imgRGB_Black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(imgRGB_Black, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                # hPts = enumerate(handLms.landmark)

                # print(hPts[8])

                for i, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    # cv2.putText(img, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

                    if i == 4:
                        cv2.circle(imgRGB_Black, (xPos, yPos), 5, (255, 0, 0), cv2.FILLED)

                    # print(i, xPos, yPos)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(imgRGB_Black, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("img", imgRGB_Black)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
