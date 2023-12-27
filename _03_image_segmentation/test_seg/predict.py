import argparse
import os
import torch
from model import UNet
from model import deeplabv3,UNet
#from data_抽检机 import transform_val
from data_切割道检测 import transform_val
import cv2
import numpy as np
import torchvision
from our1314.work.Utils import tensor2mat


def predict(opt):
    print(os.getcwd())
    path_weight = os.path.join(opt.out_path,opt.weights)
    checkpoint = torch.load(path_weight)
    net = deeplabv3()
    net.load_state_dict(checkpoint['net'])
    print("best loss:",checkpoint['loss'])
    net.eval()
    

    with torch.no_grad():
        files = [os.path.join(opt.data_path_test,f) for f in os.listdir(opt.data_path_test)]
        for image_path in files:
            src = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)   # type:cv2.Mat
            
            img, = transform_val([src])
            x = net(img.unsqueeze(0))#type:torch.Tensor
            x = x.squeeze_(dim=0)
            print("max=", torch.max(x).item(),"min=", torch.min(x).item())
            
            img = np.transpose(img.numpy(), (1, 2, 0))
            mask = np.transpose(x.numpy(), (1, 2, 0))

            tmp = img.copy()
            tmp[:,:,2:3] = tmp[:,:,2:3]*0.5 + mask*0.5

            dis = cv2.hconcat([img, cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR), tmp])
            cv2.imshow("dis", dis)
            cv2.waitKey()
        
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best_kongdong.pth', help='指定权重文件，未指定则使用官方权重！')
    parser.add_argument('--out_path', default='./run/train', type=str)  # 修改
    parser.add_argument('--data_path_test', default='D:/desktop/qgd1/test')  # 修改
    parser.add_argument('--conf', default=0.3, type=float)

    opt = parser.parse_args()

    predict(opt)
