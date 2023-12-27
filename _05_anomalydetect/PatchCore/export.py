import torch
import torchvision
from our1314 import work
from our1314.work import exportsd, importsd


# net = net_xray(False, opt.data.class_num)  # classify_net1()
# net.load_state_dict(checkpoint['net'])

net = torchvision.models.wide_resnet101_2(weights=torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
save_path = 'd:/desktop/weights/Wide_ResNet101_2_Weights.IMAGENET1K_V2.dat'

# Saving a TorchSharp format model in Python
f = open(save_path, "wb")
exportsd.save_state_dict(net.to("cpu").state_dict(), f)
f.close()
# Loading a TorchSharp format model in Python
f = open(save_path, "rb")
net.load_state_dict(importsd.load_state_dict(f))
f.close()
print('export TorchSharp model success!')
# torch.nn.Sequential