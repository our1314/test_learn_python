import argparse
import onnx
import torch.onnx.utils
from model.model import SimpleNet


def export(opt):
    path = opt.weights
    f = path.replace('.pth', '.onnx')

    input_size = (1, 3, 288, 288)
    x = torch.randn(input_size)
    checkpoint = torch.load(path)

    net = SimpleNet()
    net.load_state_dict(checkpoint['net'])
    net.eval()
    net = net.cpu()
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
    onnx.checker.check_model(onnx_model)  # check onnx model

    print('export onnx success!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./run/train_ic/best3.pth')  # 修改

    opt = parser.parse_args()
    export(opt)
