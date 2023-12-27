import cv2
import numpy as np
import halcon
from our1314.work.Utils import rad, contours2coord
from halcon.numpy_interop import himage_as_numpy_array, himage_from_numpy_array

class templateMatch():
    def __init__(self, src, coord):
        
        c1,r1,c2,r2 = coord
        model_region = halcon.gen_rectangle1(int(r1),int(c1),int(r2),int(c2))
        temp = halcon.reduce_domain(src, region=model_region)
        self.model_id = halcon.create_scaled_shape_model(temp, 5, -rad(10.0), rad(20.0), rad(0.3657), 0.8, 1/0.8, 0.01, ['none', 'no_pregeneration'], 'use_polarity', [20,50,30], 10)
        
        _, r1, c1 = halcon.area_center(model_region)
        r2, c2, phi, len1, len2 = halcon.smallest_rectangle2(model_region)

        halcon.set_shape_model_origin(self.model_id, r2[0] - r1[0], c2[0] - c1[0])
        self.select_rect = halcon.get_shape_model_contours(self.model_id, 1)
        select_pts = contours2coord(self.select_rect).T

        dis = himage_as_numpy_array(src)
        pt1 = select_pts[0]
        for pt2 in select_pts[1:]:
            cv2.line(dis, pt1, pt2, (0,0,255), 1)
            pt1 = pt2
        cv2.imshow("dis", dis)
        cv2.waitKey()
        pass

    def create(self):

        pass

    def createmodel(self):
        
        pass


    def aa(self):
        halcon.find_scaled_shape_model()
        pass


if __name__ == '__main__':
    from our1314.work.Utils import mouseSelect_simple

    path = 'd:/desktop/1.jpg'
    src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray
    a = mouseSelect_simple(src)

    coord = a.pt1[0], a.pt1[1], a.pt2[0], a.pt2[1]
    src = himage_from_numpy_array(src)
    match = templateMatch(src, coord)


    