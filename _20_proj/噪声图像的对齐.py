# var noise = torch.randn(new[] { 300, 300L }, torch.ScalarType.Float32);
#             var dis = new Mat(300, 300, MatType.CV_32F, noise.data<float>().ToArray());
#             dis = dis * 200;
#             dis = dis.ConvertScaleAbs();

import numpy as np
import cv2
from our1314.myutils.myutils import SE2,rad

img1 = np.random.randn(300,300,1)*100 + 128
img1 = img1.astype(np.uint8)

pts1 = np.array([[30,260,260,30],
                [30,30,260,260],
                [1,1,1,1]], dtype=np.float32)
H = SE2(0,0, rad(-10))
img1 = cv2.polylines(img1, [pts1[0:2,0:].T.astype(np.int32)], True, 255)


img2 = np.random.randn(300,300,1)*60 + 128
img2 = img2.astype(np.uint8)

pts2 = np.array([[30,260,260,30],
                [30,30,260,260],
                [1,1,1,1]], dtype=np.float32)
H = SE2(20,-10, rad(10))
pts = H@pts2
img2 = cv2.polylines(img2, [pts[0:2,0:].T.astype(np.int32)], True, 255)

sigma = 3
w = 2*3*sigma+1
# img1 = cv2.GaussianBlur(img1, (w,w), sigma,sigma)
# img2 = cv2.GaussianBlur(img2, (w,w), sigma,sigma)
cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.waitKey()

H = np.eye(2,3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, 5000,  1e-8)
r,H = cv2.findTransformECC(img1, img2, H, motionType=cv2.MOTION_EUCLIDEAN, criteria=criteria)

_img1 = cv2.warpAffine(img1, H, [300,300])


cv2.imshow("_img1", _img1)
# cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow('dis', cv2.hconcat([_img1,img2]))
cv2.waitKey()


