import argparse
import onnx
import torch.onnx.utils
#from data import input_size, transform_val
from data_空洞检测 import input_size, transform_val
from model import UNet,deeplabv3
from our1314.work.Utils import exportsd, importsd
import onnxruntime
from PIL import Image

def export(opt):
    path = opt.weights
    f = path.replace('.pth', '.onnx')

    size = (1, 3) + input_size
    x = torch.randn(size)
    checkpoint = torch.load(path)

    net = deeplabv3()  # classify_net1()
    net.load_state_dict(checkpoint['net'])
    net.eval()
    torch.onnx.export(net,
                      x,
                      f,
                      opset_version=11,
                      # do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      verbose='True')

    # Checks 参考 yolov7
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model,检验失败将会抛出异常！

    output_name = onnx_model.graph.output
    session = onnxruntime.InferenceSession(f, providers=['CPUExecutionProvider'])

    x = Image.open('D:/desktop/choujianji/roi/LA22089071-0152_2( 5, 1 ).jpg')
    x = transform_val([x])

    out = session.run(output_name, {'input':x})

    print('export onnx success!')

    # 导出pt,用于torchSharp
    net_ = torch.jit.trace(net, x)
    f = path.replace('.pth', '.pt')
    net_.save(f)
    print('export pt success!')

    # Saving a TorchSharp format model in Python
    save_path = path.replace('.pth', '.dat')
    f = open(save_path, "wb")
    exportsd.save_state_dict(net.to("cpu").state_dict(), f)
    f.close()
    # Loading a TorchSharp format model in Python
    f = open(save_path, "rb")
    net.load_state_dict(importsd.load_state_dict(f))
    f.close()
    print('export TorchSharp model success!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./run/train/best_kongdong_new2.pth')  # 修改

    opt = parser.parse_args()
    export(opt)
