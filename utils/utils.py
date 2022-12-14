import torch
import numpy as np
import cv2


def tensor2mat(data, dtype=np.uint8):
    """
    将给定的张量转换为Mat
    :param data:张量,三个维度，[c,h,w]
    :param dtype:模板数据类型，默认np.uint8
    :return:OpenCV Mat，三个维度，[h,w,c]
    """
    size = data.size()
    if len(size) != 3:
        assert "张量维度不为3！"
        return None
    img = data.numpy()  # type:np.ndarray
    img = img.copy()  # 没有这句会报错：Layout of the output array img is incompatible with cv::Mat
    img *= 255
    img = img.astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))  # c,h,w → h,w,c
    img = img.copy()
    return img


def drawgrid(img, size, color=(0, 0, 255), linewidth=2):
    """
    在图像上绘制指定格式的网络线
    :param img:
    :param size:
    :param color:
    :param linewidth:
    :return:
    """
    img = img.copy()
    x = np.arange(size[0]) * img.shape[1] / size[0]
    y1 = np.zeros_like(x)
    y2 = img.shape[0] * np.ones_like(x)
    p1 = np.vstack((x, y1)).T
    p2 = np.vstack((x, y2)).T

    for i in range(p1.shape[0]):
        _p1, _p2 = p1[i], p2[i]  # type:np.ndarray
        _p1 = _p1.astype(np.int)
        _p2 = _p2.astype(np.int)
        cv2.line(img, _p1, _p2, color)

    y = np.arange(size[0]) * img.shape[1] / size[0]
    x1 = np.zeros_like(x)
    x2 = img.shape[0] * np.ones_like(x)
    p1 = np.vstack((x1, y)).T
    p2 = np.vstack((x2, y)).T

    for i in range(p1.shape[0]):
        _p1, _p2 = p1[i], p2[i]  # type:np.ndarray
        _p1 = _p1.astype(np.int)
        _p2 = _p2.astype(np.int)
        cv2.line(img, _p1, _p2, color)

    return img


def rectangle(img, center, wh, color, thickness):
    """
    给定中心和宽高绘制矩阵
    :param img:
    :param center:
    :param wh:
    :param color:
    :param thickness:
    :return:
    """
    pt1 = center - wh / 2.0  # type: np.ndarray
    pt2 = center + wh / 2.0  # type: np.ndarray
    pt1 = pt1.astype(np.int)
    pt2 = pt2.astype(np.int)
    cv2.rectangle(img, pt1, pt2, color, thickness)
    return img
