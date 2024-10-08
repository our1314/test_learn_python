#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:03
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from ctpn_model import CTPN_Model
from ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from ctpn_utils import resize
import config


prob_thresh = 0.2
width = 612
device = torch.device('cpu')
weights = os.path.join(config.checkpoints_dir, 'best.dict')
img_path = './images/ocr/JPEGImages/1AM3_2023-03-09_18.00.34-235_src.png'

model = CTPN_Model()
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
model.to(device)
model.eval()


def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread(img_path)
image = resize(image, width=width)
image_c = image.copy()
h, w = image.shape[:2]
image = image.astype(np.float32) - config.IMAGE_MEAN
image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

with torch.no_grad():
    image = image.to(device)
    cls, regr = model(image)
    cls_prob = F.softmax(cls, dim=-1).cpu()#.numpy()

    tmp = cls_prob.reshape(1,32,38,10,2)
    for i in range(10):
        d = tmp[0,:,:,i,1]
        d = (d-d.min())/(d.max()-d.min())
        d = d.numpy()
        d = cv2.pyrUp(d)
        d = cv2.pyrUp(d)
        cv2.imshow("d", d)
        cv2.waitKey()
        pass
    
    regr = regr.cpu().numpy()
    anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
    bbox = bbox_transfor_inv(anchor, regr)
    bbox = clip_box(bbox, [h, w])

    fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
    select_anchor = bbox[fg, :]
    select_score = cls_prob[0, fg, 1]
    select_anchor = select_anchor.astype(np.int32)

    keep_index = filter_bbox(select_anchor, 16)

    # nsm
    select_anchor = select_anchor[keep_index]
    select_score = select_score[keep_index]
    select_score = np.reshape(select_score, (select_score.shape[0], 1))
    nmsbox = np.hstack((select_anchor, select_score))
    keep = nms(nmsbox, 0.3)
    select_anchor = select_anchor[keep]
    select_score = select_score[keep]

    # text line-
    textConn = TextProposalConnectorOriented()
    text = textConn.get_text_lines(select_anchor, select_score, [h, w])
    # print(text)

    for i in text:
        s = str(round(i[-1] * 100, 2)) + '%'
        i = [int(j) for j in i]
        cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
        cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
        cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
        cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
        cv2.putText(image_c, s, (i[0]+13, i[1]+13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2,
                    cv2.LINE_AA)

    dis(image_c)
