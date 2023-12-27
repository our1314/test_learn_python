import cv2

# 使用 cv2.imdecode 函数读取中文路径图片
import cv2
import numpy as np

img = cv2.imdecode(np.fromfile("微信图片.jpg", dtype=np.uint8), -1)
img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
cv2.imshow("Img", img)
cv2.waitKey(0)
