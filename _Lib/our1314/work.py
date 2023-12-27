from datetime import datetime
import os
import cv2
import numpy as np
from PIL import Image
from math import *
from torchvision.transforms import functional as F
import random
import torchvision
import torch
import torch
import leb128
from collections import OrderedDict

#region math
def SO2(rad):
    r = np.array([
            [cos(rad),-sin(rad)],
            [sin(rad),cos(rad)],
        ])
    return r

def _SO2(R):
    cos_theta = R[0,0]
    sin_theta = R[1,0]
    theta = atan2(cos_theta, sin_theta)
    return theta

def SE2(x,y,rad):
    r = np.array([
            [cos(rad),-sin(rad),x],
            [sin(rad),cos(rad),y],
            [0,0,1]
        ])
    return r

def _SE2(H):
    x,y = H[0,2],H[1,2]
    cos_theta = H[0,0]
    sin_theta = H[1,0]
    theta = atan2(cos_theta, sin_theta)
    return x,y,theta

#绕3D坐标系X轴旋转的旋转矩阵
def RX(rad):
    r = np.array([
            [1,0,0],
            [0,cos(rad),-sin(rad)],
            [0,sin(rad),cos(rad)]
        ])
    return r
#绕3D坐标系X轴旋转的旋转矩阵
def RY(rad):
    r = np.array([
            [cos(rad),0,sin(rad)],
            [0,1,0],
            [-sin(rad),0,cos(rad)]
        ])
    return r
#绕3D坐标系X轴旋转的旋转矩阵
def RZ(rad):
    r = np.array([
            [cos(rad),-sin(rad),0],
            [sin(rad),cos(rad),0],
            [0,0,1]
        ])
    return r

def Pxyz(x,y,z):
    H = np.eye(4)
    H[0,3],H[1,3],H[2,3]=x,y,z
    return H

def Homogeneous(m):
    h,w = m.shape
    m = np.column_stack([m,np.zeros(h,1)])
    m = np.row_stack([m,np.zeros(1,w+1)])
    m[-1,-1]=1
    return m

def SE3(px,py,pz,rx,ry,rz):
    Rx = Homogeneous(RX(rx))
    Ry = Homogeneous(RY(ry))
    Rz = Homogeneous(RZ(rz))
    P = Pxyz(px,py,pz)
    H = P@Rz@Ry@Rx
    return H

def _SE3(H):
    pass


def deg(rad):
    return rad*180/pi

def rad(deg):
    return deg*pi/180
#endregion

#region utils
def pil2mat(image):
    mat = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return mat

def mat2pil(mat):
    image = Image.fromarray(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))
    return image

def tensor2mat(data, dtype=None):
    """
    将给定的张量转换为Mat类型图像，并自动*255，并交换通道RGB→BGR
    :param data:张量,三个维度，[c,h,w]
    :param dtype:模板数据类型，默认np.uint8
    :return:OpenCV Mat，三个维度，[h,w,c]
    """
    assert len(data.shape)==3 , "张量维度不为3！"

    img = data.detach().numpy()  # type:np.ndarray
    img = img.copy()  # 没有这句会报错：Layout of the output array img is incompatible with cv::Mat
    img = np.transpose(img, (1, 2, 0))  # c,h,w → h,w,c
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def mat2tensor(img:np.array, dtype=np.uint8):
    """
    输入图像数据为0-255，默认为BGR通道，自动交换为RGB通道，归一化至0-1
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = F.ToTensor(img)
    return tensor

def drawgrid(img, size, color=(0, 0, 255), linewidth=1):
    """
    在图像上绘制指定格式的网络线
    :param img:
    :param size:
    :param color:
    :param linewidth:
    :return:
    """
    dis = img.copy()
    x = np.arange(size[0]) * dis.shape[1] / size[0]
    y1 = np.zeros_like(x)
    y2 = dis.shape[0] * np.ones_like(x)
    p1 = np.vstack((x, y1)).T
    p2 = np.vstack((x, y2)).T

    for i in range(p1.shape[0]):
        _p1, _p2 = p1[i], p2[i]  # type:np.ndarray
        _p1 = _p1.astype(np.int)
        _p2 = _p2.astype(np.int)
        cv2.line(dis, _p1, _p2, color)

    y = np.arange(size[0]) * dis.shape[1] / size[0]
    x1 = np.zeros_like(x)
    x2 = dis.shape[0] * np.ones_like(x)
    p1 = np.vstack((x1, y)).T
    p2 = np.vstack((x2, y)).T

    for i in range(p1.shape[0]):
        _p1, _p2 = p1[i], p2[i]  # type:np.ndarray
        _p1 = _p1.astype(np.int)
        _p2 = _p2.astype(np.int)
        cv2.line(dis, _p1, _p2, color)

    return dis

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

# 获取按时间排序的最后一个文件
def getlastfile(path, ext='.pth'):
    if os.path.exists(path) is not True: return None
    list_file = [path + '/' + f for f in os.listdir(path) if f.endswith(ext)]  # 列表解析
    if len(list_file) > 0:
        list_file.sort(key=lambda fn: os.path.getmtime(fn))
        return list_file[-1]
    else:
        return None

def yolostr2data(yolostr: str):
    """
    解析yolo字符串，转换为np.ndarray
    """
    data = []
    yolostr = yolostr.strip()
    arr = yolostr.split('\n')
    arr = [f.strip() for f in arr]
    arr = [f for f in arr if f != ""]

    for s in arr:
        a = s.split(' ')
        a = [f.strip() for f in a]
        a = [f for f in a if f != ""]
        data.append((int(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])))
    return np.array(data)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def addWeightedMask(src1, alpha, mask, beta, blendChannle=2):
    src_mask = cv2.copyTo(src1, mask=mask)
    src_mask[:,:,blendChannle:blendChannle+1] = src_mask[:,:,blendChannle:blendChannle+1] * alpha + mask * beta
    src1 = cv2.copyTo(src_mask, mask=mask, dst=src1)
    return src1

def Now():
    return datetime.now().strftime("%Y-%m-%d_%H.%M.%S-%f")

def GetAllFiles(path:str):
    '''
    获取当前目录，及子目录下的所有文件的路径。
    '''
    files_all = []
    for root,dirs,files in os.walk(path):
        for file in files:
            files_all.append(os.path.join(root,file))
    return files_all
#endregion

#region ext_transform
class ToTensors:
    def __call__(self, imgs):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = []
        for x in imgs:
            if isinstance(x, Image.Image):
                img = F.to_tensor(x)
                result.append(img)

            elif isinstance(x, np.ndarray):
                img = F.to_tensor(x)
                result.append(img)
            elif isinstance(x, torch.Tensor):
                result.append(x)

            else:
                assert '数据类型应该是张量或者ndarray'

        return result

# 按比例将长边缩放至目标尺寸
class Resize1:
    def __init__(self, width):
        self.width = width

    def __call__(self, imgs):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = []
        for x in imgs:
            if isinstance(x, torch.Tensor):
                h, w = x.shape[1],x.shape[2]
                scale = self.width / max(w, h)
                W, H = round(w * scale), round(h * scale)
                img = F.resize(x,[H,W], antialias=True)#antialias=True避免输出警告
                result.append(img)

            elif isinstance(x, np.ndarray):
                h, w = x.shape[0],x.shape[1]
                scale = self.width / max(w, h)
                W, H = round(scale * w), round(scale * h)
                result.append(cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR))

            else:
                assert '数据类型应该是张量或者ndarray'

        return result


class PadSquare:
    def __call__(self, imgs):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = []
        for x in imgs:
            if isinstance(x, torch.Tensor):
                h, w = x.shape[1],x.shape[2]
                width = max(w, h)
                pad_left = round((width - w) / 2.0)
                pad_right = width - w - pad_left
                pad_up = round((width - h) / 2.0)
                pad_down = width - h - pad_up
                img = F.pad(x, [pad_left, pad_up, pad_right, pad_down])
                result.append(img)

            elif isinstance(x, np.ndarray):
                h, w = x.shape[0],x.shape[1]
                width = max(w, h)
                pad_left = round((width - w) / 2.0)
                pad_right = width - w - pad_left
                pad_up = round((width - h) / 2.0)
                pad_down = width - h - pad_up

                result.append(cv2.copyMakeBorder(x, pad_up, pad_down, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0))

            else:
                assert '数据类型应该是张量或者ndarray'

        return result

class randomaffine_imgs:
    def __init__(self, p:float, rotate:list[float], transx:list[float], transy:list[float], scale:list[float]):
        self.p = p
        self.rotate = rotate
        self.transx = transx
        self.transy = transy
        self.scale = scale
        

    def __call__(self, imgs:list):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = imgs.copy()
        value = random.uniform(0,1)
        if value < self.p:
            rot_deg = 0 if self.rotate == None else random.uniform(self.rotate[0], self.rotate[1])
            transx = 0 if self.transx == None else random.uniform(self.transx[0], self.transx[1])
            transy = 0 if self.transy == None else random.uniform(self.transy[0], self.transy[1])
            scale = 0 if self.scale == None else random.uniform(min(self.scale), max(self.scale))

            result = []
            for x in imgs:
                if isinstance(x, torch.Tensor):
                    h, w = x.shape[1],x.shape[2]
                    img_trans = F.affine(x, rot_deg, [int(transx*w),int(transy*h)], scale, 1, interpolation=F.InterpolationMode.BILINEAR)
                    result.append(img_trans)
                
                elif isinstance(x, np.ndarray):
                    h,w = x.shape[0],x.shape[1]
                    angle_rad = rad(rot_deg)
                    H1 = np.array([
                        [1,0,-w/2],
                        [0,1,-h/2],
                        [0,0,1]
                    ])
                    H2 = np.array([
                        [scale*cos(angle_rad),-sin(angle_rad),transx*w],
                        [sin(angle_rad),scale*cos(angle_rad),transy*h],
                        [0,0,1]
                    ])
                    H = np.linalg.inv(H1)@H2@H1
                    img_trans = cv2.warpAffine(x, H[0:2,0:3], (w,h))
                    result.append(img_trans)

                else:
                    assert '数据类型应该是张量或者ndarray'
        return result

class randomvflip_imgs:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs:list):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = imgs.copy()
        value = random.uniform(0,1)
        if value < self.p:
            for i,x in enumerate(imgs):
                if isinstance(x, torch.Tensor):
                    result[i]=(F.vflip(x))
                elif isinstance(x, np.ndarray):
                    result[i]=cv2.flip(x,0)
                else:
                    assert '数据类型应该是张量或者ndarray'
        return result
    
class randomhflip_imgs:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs:list):
        assert type(imgs) == list,'类型不为list'
        assert len(imgs) !=0, '数量为0'

        result = imgs.copy()
        value = random.uniform(0,1)
        if value < self.p:
            for i,x in enumerate(imgs):
                if isinstance(x, torch.Tensor):
                    result[i]=F.hflip(x)
                elif isinstance(x, np.ndarray):
                    result[i]=cv2.flip(x,1)
                else:
                    assert '数据类型应该是张量或者ndarray'
        return result


#endregion

#region export for torchsharp
class importsd():
    _DTYPE_SIZE_MAP = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float16: 2,
        np.float32: 4,
        np.float64: 8,
    }


    def _get_elem_type(type_num: int):
        if type_num == 0:
            return np.uint8
        elif type_num == 1:
            return np.int8
        elif type_num == 2:
            return np.int16
        elif type_num == 3:
            return np.int32
        elif type_num == 4:
            return np.int64
        elif type_num == 5:
            return np.float16
        elif type_num == 6:
            return np.float32
        elif type_num == 7:
            return np.float64
        elif type_num == 11:
            # return torch.bool
            raise NotImplemented("Unsupported data type")
        elif type_num == 15:
            # return torch.bfloat16
            raise NotImplemented("Unsupported data type")
        elif type_num == 4711:
            raise NotImplemented("Unsupported data type")
        else:
            raise ValueError("cannot decode the data type")


    def load_state_dict(stream):
        """
        Loads a PyTorch state dictionary using the format that saved by TorchSharp.
        :param stream: An write stream opened for binary I/O.
        :return sd: A dictionary can be loaded by 'model.load_state_dict()'
        """
        sd = OrderedDict()
        dict_len, _ = leb128.u.decode_reader(stream)
        for i in range(dict_len):
            key_len, _ = leb128.u.decode_reader(stream)
            key_name = stream.read(key_len).decode("utf-8")

            ele_type, _ = leb128.u.decode_reader(stream)
            buffer_dtype = importsd._get_elem_type(ele_type)

            buffer_shape_len, _ = leb128.u.decode_reader(stream)
            buffer_shape = tuple(leb128.u.decode_reader(stream)[0] for _ in range(buffer_shape_len))
            if buffer_shape:
                data_size = np.prod(buffer_shape)
            else:
                data_size = 1

            data_size_bytes = data_size * importsd._DTYPE_SIZE_MAP[buffer_dtype]
            sd[key_name] = torch.from_numpy(
                np.frombuffer(
                    stream.read(data_size_bytes), dtype=buffer_dtype, count=data_size
                ).reshape(buffer_shape)
            )
        return sd
    
class exportsd():
    def _elem_type(t):
        dt = t.dtype

        if dt == torch.uint8:
            return 0
        elif dt == torch.int8:
            return 1
        elif dt == torch.int16:
            return 2
        elif dt == torch.int32:
            return 3
        elif dt == torch.int64:
            return 4
        elif dt == torch.float16:
            return 5
        elif dt == torch.float32:
            return 6
        elif dt == torch.float64:
            return 7
        elif dt == torch.bool:
            return 11
        elif dt == torch.bfloat16:
            return 15
        else:
            return 4711

    def _write_tensor(t, stream):
        stream.write(leb128.u.encode(exportsd._elem_type(t)))
        stream.write(leb128.u.encode(len(t.shape)))
        for s in t.shape:
            stream.write(leb128.u.encode(s))
        stream.write(t.numpy().tobytes())

    def save_state_dict(sd, stream):
        """
        Saves a PyToch state dictionary using the format that TorchSharp can
        read.

        :param sd: A dictionary produced by 'model.state_dict()'
        :param stream: An write stream opened for binary I/O.
        """
        stream.write(leb128.u.encode(len(sd)))
        for entry in sd:
            stream.write(leb128.u.encode(len(entry)))
            stream.write(bytes(entry, 'utf-8'))
            exportsd._write_tensor(sd[entry], stream)
#endregion

#region mouseSelect
class mouseSelect_simple():
    def __init__(self, src, windowName='dis'):
        self.src = src
        self.windowName = windowName
        self.down = False
        
        cv2.namedWindow(windowName)
        cv2.setMouseCallback(windowName, self.onmouse)
        cv2.imshow(windowName, src)
        cv2.waitKey()

    def onmouse(self, *p):
        event, x, y, flags, param = p   
        if event == cv2.EVENT_LBUTTONDOWN:
            self.down = True
            self.pt1 = np.array([x,y])

        if event == cv2.EVENT_MOUSEMOVE and self.down==True:
            self.pt2 = np.array([x,y])
            dis = self.src.copy()
            cv2.rectangle(dis, self.pt1, self.pt2, (0,0,255), 2)
            cv2.imshow(self.windowName , dis)

        if event == cv2.EVENT_LBUTTONUP:
            self.down = False
            self.pt2 = np.array([x,y])
            return self.pt1, self.pt2
        
        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.waitKey(200)
            cv2.destroyWindow(self.windowName)
#endregion


if __name__ == '__main__':
    transform1 = torchvision.transforms.Compose([
            ToTensors(),
            Resize1(448),#等比例缩放
            PadSquare(),
            randomaffine_imgs(1, [-0,0], [-0,0], [-0,0], [1,1/1]),
            randomvflip_imgs(0.5),
            randomhflip_imgs(0.5)
        ])
    
    data_path = 'D:/desktop/choujianji/roi/mask/train'
    Images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')]
    Labels = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')]
    for i in range(len(Images)):
        # image = cv2.imdecode(np.fromfile(Images[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED) # type:cv2.Mat
        # label = cv2.imdecode(np.fromfile(Labels[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED) # type:cv2.Mat

        image = Image.open(Images[i])
        label = Image.open(Labels[i])
        #b1 = transform1([image])[0]

        b1,b2 = transform1([image,label])
        if isinstance(b1, np.ndarray):
            b1 = b1/255.0
            b2 = b2/255.0

            b1[:,:,2] += 0.6*b2
            dis = b1
            a1 = np.max(dis)
            pass
        elif isinstance(b1, torch.Tensor):
            m1 = torch.max(b1)
            b1[2:,:,:] += 0.6*b2
            dis = b1.numpy()
            dis = np.transpose(dis, [1,2,0])
            a1 = np.max(dis)
            a1 = np.max(dis)
            pass
        #F.to_pil_image(dis).show()
        cv2.imshow('dis', dis)
        cv2.waitKey()
    # x = torch.rand((3,300,300),dtype=torch.float)
    # y = tensor2mat(x)
    # y[y>0]=0
    # y=drawgrid(y, [10,10])
    # cv2.imshow('dis', y)
    # cv2.waitKey()
    # pass

    src1 = np.random.rand(256,256,3)*255
    mask = np.random.rand(256,256,1)*255

    src1 = src1.astype(np.uint8)
    mask = mask.astype(np.uint8)

    dis = addWeightedMask(src1,0.5,mask,0.5)
    cv2.imshow("dis",dis)
    cv2.waitKey()

