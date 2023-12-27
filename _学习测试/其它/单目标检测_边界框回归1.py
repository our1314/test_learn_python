import os
import keras.optimizer_v2.adam
import keras.optimizers
from keras import models
from keras import layers
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet101V2
import cv2
import numpy as np
import csv

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 让GPU失效

from keras import backend as K


def my_loss(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    ##intersection = K.sum(y_truef * y_predf)
    err = np.linalg.norm(y_truef - y_predf)
    return err


"""
https://www.jianshu.com/p/9eff60847407
"""

"""
思路：
1、利用OpenCV创建数据集
2、
"""
scale = 8
target_size = (int(2048 / scale), int(2448 / scale))  # 行,列
h = target_size[0]
w = target_size[1]

IS_CREATE_DATASET = False

if IS_CREATE_DATASET:
    # region 1、创建数据集
    """
    1、读取模板
    2、随机读取背景
    3、将模板贴到背景上
    """

    dataset_count = 300
    template = cv2.imread('dataset\\create_dataset\\template.png')
    roi = template[517: 1237, 520: 1387]
    roi_shape = np.shape(roi)
    img_shape = np.shape(template)

    # cv2.imshow('roi', roi)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    background_files = os.listdir('dataset\\create_dataset\\background')
    background_count = len(background_files)
    for i in np.arange(1, dataset_count + 1, 1):
        index = np.random.randint(0, background_count - 1)
        background = cv2.imread('dataset\\create_dataset\\background\\' + background_files[index])

        r0 = np.random.randint(0, img_shape[0] - roi_shape[0] + 1)
        r1 = r0 + roi_shape[0]
        c0 = np.random.randint(1, img_shape[1] - roi_shape[1] + 1)
        c1 = c0 + roi_shape[1]

        background[r0:r1, c0:c1] = roi

        # cv2.imshow("1", background)
        # cv2.waitKey()

        background = cv2.resize(background, (w, h))

        img_name = str(i) + '.png'
        cv2.imwrite('dataset\\dataset\\' + img_name, background)

        with open("dataset\\coord.csv", 'a', newline='', encoding='utf8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([img_name, r0 / scale, r1 / scale, c0 / scale, c1 / scale])
    # endregion

# region 2、准备数据集
BASE_PATH = 'dataset'
IMAGE_PATH = os.path.join(BASE_PATH, 'dataset')
ANNOTS_PATH = os.path.join(BASE_PATH, 'coord.csv')

data = []
targets = []
with open(ANNOTS_PATH) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        (filename, r0, r1, c0, c1) = row
        img = cv2.imread('dataset\\dataset\\' + filename)
        data.append(img)

        r0 = float(r0) / h
        r1 = float(r1) / h
        c0 = float(c0) / w
        c1 = float(c1) / w
        targets.append((r0, r1, c0, c1))

data = np.array(data, dtype='float32') / 255.0
targets = np.array(targets, dtype="float32")

train_img = data[0:200]
val_img = data[200:260]
test_img = data[260:300]

train_label = targets[0:200]
val_label = targets[200:260]
test_label = targets[260:300]

# import cv2
#
# for i in np.arange(len(train_img)):
#     img = train_img[i]
#     label = train_label[i]
#
#     img = img * 256
#     img = cv2.convertScaleAbs(img)
#
#     pt1 = (int(label[2]*w), int(label[0]*h))
#     pt2 = (int(label[3]*w), int(label[1]*h))
#     cv2.rectangle(img, pt1, pt2, (0, 0, 255))
#     cv2.imshow("1", img)
#     cv2.waitKey()
#     pass
# endregion


model_path = 'dataset\\目标检测_边界回归.h5'
net = models.Sequential()

if not os.path.exists(model_path):
    # region 3、创建网络
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=target_size + (3,))
    base_model.trainable = False

    net.add(base_model)
    net.add(layer=layers.Flatten())
    net.add(layer=layers.Dense(512, activation='relu'))
    net.add(layer=layers.Dense(256, activation='relu'))
    net.add(layer=layers.Dense(128, activation='relu'))
    net.add(layer=layers.Dense(64, activation='relu'))
    net.add(layer=layers.Dense(32, activation='relu'))
    net.add(layer=layers.Dense(4, activation='sigmoid'))

    print(net.summary())
    print(net.output)

    # net.compile(loss='mae', optimizer=keras.optimizers.rmsprop_v2.RMSProp(learning_rate=1e-3))
    net.compile(loss='mse', optimizer=keras.optimizer_v2.adam.Adam(learning_rate=1e-6))
    # net.compile(loss='mae', optimizer=keras.optimizers.sgd_experimental.SGD(learning_rate=1e-2))
    # net.compile(loss='mae', optimizer=my_loss)
    net.fit(x=train_img, y=train_label, batch_size=8, epochs=500, validation_data=(val_img, val_label), verbose=1)
    pass

    net.save('dataset\\目标检测_边界回归.h5')
    # endregion
else:
    net = models.load_model(model_path)

# for img in train_img:
#     loc = net.predict(img)
#     loc = loc * target_size
#     print(loc)
#
#     img = img * 256
#     img = np.astype(img, np.uint8)
#     cv2.imshow('1', img)
#     cv2.waitKey()

for i in np.arange(len(train_img)):
    label = train_label[0]
    img = train_img[i]
    data = np.expand_dims(img, 0)
    loc = net.predict(data)
    loc = np.squeeze(loc, 0)
    # loc = (loc[0:2]*h).append(loc[2:4] * w)

    err = np.linalg.norm(loc - label)
    print(err)
    # label = train_label[i]

    img = img * 256
    img = cv2.convertScaleAbs(img)

    pt1 = (int(loc[2] * w), int(loc[0] * h))
    pt2 = (int(loc[3] * w), int(loc[1] * h))
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
    cv2.imshow("1", img)
    cv2.waitKey()
    pass
