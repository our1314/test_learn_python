import torch
import math
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights
import torch.nn.functional as F
from torchvision import models
from .projection import Projection
from .patchmaker import PatchMaker
from .common import NetworkFeatureAggregator,RescaleSegmentor
from .Preprocessing import Preprocessing
from .Aggregator import Aggregator
from .discriminator import Discriminator
import numpy as np
import cv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_BACKBONES = {
    "cait_s24_224" : "cait.cait_S24_224(True)",
    "cait_xs24": "cait.cait_XS24(True)",
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet18": "models.resnet18(pretrained=True)",
    "resnet50": "models.resnet50(pretrained=True)",
    "mc3_resnet50": "load_mc3_rn50()", 
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)",
    "ref_wideresnet50": "load_ref_wrn50()",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #0、参数
        self.input_shape = (3,288,288)
        self.pre_proj = 1
        self.layers_to_extract_from = ('layer2','layer3')
        self.train_backbone = False
        self.mix_noise = 1
        self.noise_std = 0.015#0.015

        #1、主干网
        self.backbone = eval(_BACKBONES['wideresnet50'])
        self.backbone.to(self.device)
        
        self.patch_maker = PatchMaker(patchsize=3, stride=1)

        #2、特征提取和聚合模块(有三个子模块：主干网、预处理器、特征聚合器)
        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = NetworkFeatureAggregator(self.backbone, self.layers_to_extract_from, self.device, train_backbone=False)
        feature_dimensions = feature_aggregator.feature_dimensions(self.input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator#特征预适应聚合器

        preprocessing = Preprocessing(feature_dimensions, output_dim=1536)
        self.forward_modules["preprocessing"] = preprocessing#预处理器

        self.target_embed_dimension = 1536
        preadapt_aggregator = Aggregator(target_dim=self.target_embed_dimension)
        preadapt_aggregator.to(device=self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator#预适应聚合器

        #3、判别器模块
        self.th = 0.5#判别器阈值
        self.lr = 0.0002
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=2, hidden=1024)
        self.discriminator.to(self.device)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=1e-5)

        #
        self.pre_proj = 1
        if self.pre_proj > 0:#Projection只有一层线性层
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, self.pre_proj, layer_type=0)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), self.lr*0.1)

        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), self.lr)

        self.anomaly_segmentor = RescaleSegmentor(device=self.device, target_size=self.input_shape[-2:])


    def forward(self, images, train=True):
        #images = images.to(self.device)

        #1、提取特征
        self.forward_modules.eval()#特征提取模块改为评估模式
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        if self.pre_proj > 0:
            true_feats = self.pre_projection(self._embed(images, evaluation=False)[0])#提取特征+适配特征(自适应平均池化+Stack)+pre_projection 10368,1536
        else:
            true_feats = self._embed(images, evaluation=False)[0]#提取特征+适配特征
        
        #2、生成噪声，并添加到正样本#（优化方案：判断添加噪声的样本与未添加时的相似度，如相似度超过某一阈值则删除。）
        noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(images.device) # (N, K)
        shape = noise_one_hot.shape
        noise = torch.stack([torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape) for k in range(self.mix_noise)], dim=1).to(images.device) # (N, K, C)
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        fake_feats = true_feats + noise#给正样本的特征添加噪声得到负样本。
        
        #3、判别器进行判断
        scores = self.discriminator(torch.cat([true_feats, fake_feats]))#判别器
        true_scores = scores[:len(true_feats)]
        fake_scores = scores[len(fake_feats):]

        th = self.th
        p_true = (true_scores.detach() >= th).sum() / len(true_scores)
        p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
        true_loss = torch.clip(-true_scores + th, min=0)
        fake_loss = torch.clip(fake_scores + th, min=0)
        
        loss = true_loss.mean() + fake_loss.mean()

        self.proj_opt.zero_grad()
        #self.backbone_opt.zero_grad()
        self.disc_opt.zero_grad()

        loss.backward()

        if self.pre_proj > 0:
            self.proj_opt.step()

        if self.train_backbone:
            self.backbone_opt.step()
        
        self.disc_opt.step()
        loss = loss.detach().cpu().item()
        
        return loss,p_true.cpu().item(),p_fake.cpu().item()
           

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""

        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
        #至此使用resnet主干网提取了layer2、layer3特征

        #从特征图提取patch
        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]#[bs,512,36,36]→[bs,1296,512,3,3]   [bs,1024,18,18]→[bs,324,1024,3,3]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:])#[16, 18, 18, 1024, 3, 3]
            _features = _features.permute(0, -3, -2, -1, 1, 2)#[16, 1024, 3, 3, 18, 18]
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])#147456,18,18
            _features = F.interpolate(_features.unsqueeze(1),size=(ref_num_patches[0],ref_num_patches[1]), mode="bilinear",align_corners=False,)#对特征图进行双线性插值[147456, 1, 36, 36]
            _features = _features.squeeze(1)#[147456, 36, 36]
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])#[16, 1024, 3, 3, 36, 36])
            _features = _features.permute(0, -2, -1, 1, 2, 3)#[16, 36, 36, 1024, 3, 3]
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])#[16, 1296, 1024, 3, 3]
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]#20736,512,3,3    20736,1024,3,3
        
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features) # pooling each feature to same channel and stack together  自适应平均池化并合并特征，没有可学习参数
        features = self.forward_modules["preadapt_aggregator"](features) # further pooling  自适应平均池化，没有可学习参数


        return features, patch_shapes


    def predict(self, images):
        """Infer score and mask for a batch of images."""
        #images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features, patch_shapes = self._embed(images,provide_patch_shapes=True,evaluation=True)#[1296, 1536] [36,36] [18,18]
            if self.pre_proj > 0:
                features = self.pre_projection(features)#[1296, 1536]

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = -self.discriminator(features)#[1296, 1]
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)#(1, 1296, 1)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)#(1, 1296, 1)
            image_scores = self.patch_maker.score(image_scores)#(1,) [4.389655]

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales = patch_shapes[0]#[36, 36]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])#(1, 36, 36)
            features = features.reshape(batchsize, scales[0], scales[1], -1)#[1, 36, 36, 1536]
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)
        return masks
        #return list(image_scores), list(masks), list(features)


if __name__ == "__main__":
    print(_BACKBONES['wideresnet50'])
    net = eval(_BACKBONES['wideresnet50'])
    print(net)
    pass