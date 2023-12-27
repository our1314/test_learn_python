from typing import Any
import cv2
import numpy as np
import onnxruntime as ort
import torch
from our1314.myutils.myutils import sigmoid
import os
import time

class yolo:
    def __init__(self, onnx_path, confidence_thres=0.1, iou_thres=0):
        self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape

        self.input_name = model_inputs[0].name
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
    
    def __call__(self, img):
        image_data = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: image_data})
        output_img = self.postprocess(img, outputs)
        return output_img
    
    def getroi(self, img):
        _, mask = self.__call__(img.copy())
        x,y,w,h = cv2.boundingRect(mask)
        roi = cv2.copyTo(img, mask)
        x1,y1,x2,y2 = x,y,x+w,y+h
        roi = roi[y1:y2, x1:x2]
        return roi

    #预处理
    def preprocess(self, input_image):
        img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #region pad
        img_height, img_width = input_image.shape[:2]
        scale = 640.0 / max(img_height,img_width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        self.up, self.left = (640-img.shape[0])//2, (640-img.shape[1])//2
        down = 640 - img.shape[0] - self.up
        right = 640 - img.shape[1] - self.left

        img = cv2.copyMakeBorder(img, self.up, down, self.left, right, cv2.BORDER_CONSTANT, value=(128,128,128))
        #endregion

        image_data = np.array(img) / 255.0
        #image_data = (image_data - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])

        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data
    
    #后处理
    def postprocess(self, input_image, output):
        outputs = np.transpose(np.squeeze(output[0]))
        proto = output[1]
        #print(outputs[0])
        
        rows = outputs.shape[0]
        
        boxes = []
        scores = []
        class_ids = []

        img_height, img_width = input_image.shape[:2]
        
        scale = 640.0/(max(img_height, img_width))
        index = []
        for i in range(rows):
            rrr = outputs[i]
            classes_scores = outputs[i][4:5]
            max_score = np.amax(classes_scores)
            
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                left = (x-self.left-w/2)//scale
                top = (y-self.up-h/2)//scale
                width = w//scale
                height = h//scale

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                index.append(i)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

            ##########################################
            idx = index[i]
            r = outputs[idx]
            masks_in = r[5:]
            masks_in = masks_in[None,]

            box = r[:4]
            box = np.array(box)
            box = box[None,]

            masks = self.process_mask(proto[0], masks_in, box, [640,640], True)
            masks = masks[self.up:640-self.up, 0:]
            masks = cv2.resize(masks, (img_width, img_height))

            aaa = cv2.merge([np.zeros_like(masks), np.zeros_like(masks), masks])
            dis = cv2.addWeighted(input_image, 1, aaa, 0.6, 0)

            # cv2.imshow("dis", dis)
            # cv2.moveWindow("dis",0,0)
            # cv2.waitKey()
            # pass

        # Return the modified input image
        return input_image, masks

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0,255,255), 2)
        # label = 'a'
        # (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # label_x = x1
        # label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        # cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), (0,255,255), cv2.FILLED)
        # cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape

        masks = (masks_in @ protos.reshape(32,-1))
        masks = sigmoid(masks)
        masks = masks.reshape(-1, mh, mw)
        
        downsampled_bboxes = self.wywh2xyxy(bboxes)

        #将box坐标缩放至mask图尺寸
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        a = masks > 0.5
        
        masks = a.astype(np.uint8)
        masks = np.transpose(masks, (1,2,0))
        masks = cv2.resize(masks, shape, interpolation=cv2.INTER_BITS2)
        masks *= 255
        # cv2.imshow("dis", masks*255)
        # cv2.moveWindow("dis",0,0)
        # cv2.waitKey()
        # if upsample:
        #     masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        return masks
    
    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1,y1,x2,y2 = x1[None,None,], y1[None,None,], x2[None,None,], y2[None,None,]
        r = np.arange(w)[None, None, :]  # rows shape(1,w,1)
        c = np.arange(h)[None, :, None]  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def wywh2xyxy(self, box_xywh):
        box_xyxy = box_xywh.copy()
        box_xyxy[:, 0] = box_xywh[:,0]-box_xywh[:,2]/2
        box_xyxy[:, 1] = box_xywh[:,1]-box_xywh[:,3]/2
        box_xyxy[:, 2] = box_xywh[:,0]+box_xywh[:,2]/2
        box_xyxy[:, 3] = box_xywh[:,1]+box_xywh[:,3]/2
        return box_xyxy

if __name__ == "__main__":
    seg = yolo('yolov8-seg.onnx') 
    src = cv2.imdecode(np.fromfile('D:/work/files/deeplearn_datasets/其它数据集/抽检机缺陷检测/IW8500JG/1.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)

    start = time.time()
    dis,mask = seg(src)
    end = time.time()
    print(f'运行时间：{end-start}')

    mask = cv2.merge([np.zeros_like(mask), np.zeros_like(mask), mask])
    dis = cv2.addWeighted(dis, 1, mask, 0.6, 0)
    cv2.imshow("dis", dis)
    cv2.waitKey()
    
    roi1 = seg.getroi(src)
    src = cv2.imdecode(np.fromfile('D:/work/files/deeplearn_datasets/其它数据集/抽检机缺陷检测/2KG023075JL-155/2.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)
    roi2 = seg.getroi(src)

    roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    H = np.eye(2,3,dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-8)
    cc, H = cv2.findTransformECC(roi1, roi2, H, cv2.MOTION_EUCLIDEAN, criteria)
    troi1 = cv2.warpAffine(roi1, H, (roi2.shape[1],roi2.shape[0]))

    path = 'D:/desktop/test_roi'
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(f'{path}/src1.png', roi1)
    cv2.imwrite(f'{path}/src2.png', roi2)

    cv2.imwrite(f'{path}/1.png', troi1)
    cv2.imwrite(f'{path}/2.png', roi2)
    
    cv2.imshow("dis", cv2.hconcat([troi1, roi2]))
    cv2.waitKey()