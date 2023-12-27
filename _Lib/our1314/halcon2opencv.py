import halcon
from halcon.numpy_interop import himage_from_numpy_array, himage_as_numpy_array
import numpy as np
import cv2

def ndarray2hobject(mat):
    return himage_from_numpy_array(mat)

def hobject2ndarray(hobj):
    return himage_as_numpy_array(hobj)

def contours2coord(contours):
    coord=[]
    for i in range(halcon.count_obj(contours)):
        objectseleted = halcon.select_obj(contours, i+1)
        row, col = halcon.get_contour_xld(objectseleted)

        for r,c in zip(row, col):
            coord.append([c,r])
    return np.array(coord).T.astype(np.int32)

if __name__ == '__main__':
    img = cv2.imread('d:/desktop/tmp.png', cv2.IMREAD_COLOR)
    hobj = ndarray2hobject(img)
    w, h = halcon.get_image_size(hobj)