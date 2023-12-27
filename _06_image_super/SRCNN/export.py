import argparse
import onnx
import torch.onnx.utils
from image_classification.cnn_imgcls.data import data_xray_sot23, data_xray_sc88, data_xray_sc70, data_xray_sc89, \
    data_xray_sod123, data_xray_sod323, data_xray_sot23_juanpan, data_xray_sod523, data_xray_sod723, data_xray_sot25, \
    data_xray_sot26, data_xray_sot23e, data_oqa_chr, data_oqa_agl, data_cleaner
from image_classification.cnn_imgcls.models.net_xray import net_xray


def export(opt):
    path = opt.weights
    f = path.replace('.pth', '.onnx')

    input_size = (1, 3) + opt.data.input_size
    x = torch.randn(input_size)
    checkpoint = torch.load(path)

    net = net_xray(False, opt.data.class_num)  # classify_net1()
    net.load_state_dict(checkpoint['net'])
    net.eval()
    torch.onnx.export(net,
                      x,
                      f,
                      opset_version=10,
                      # do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      verbose='True')

    # Checks 参考 yolov7
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    print('export success!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./run/train/cleaner/weights/best.pth')  # 修改
    parser.add_argument('--data', default=data_cleaner, type=dict)  # 修改

    opt = parser.parse_args()
    export(opt)
