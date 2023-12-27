from collections import OrderedDict
import torch
from torch import nn
import torchvision
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F


class backbone_pretrain(nn.Module):
    def __init__(self):
        super(backbone_pretrain,self).__init__()

        wideresnet = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        return_layers = {'layer2':'layer2','layer3':'layer3'}
        self.net = IntermediateLayerGetter(wideresnet, return_layers)

    def forward(self, x):
        x = self.net(x)
        return x['layer2'], x['layer3']

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.backbone = backbone_pretrain()
        self.backbone.eval()

        self.project = nn.Linear(1536,1536)
        self.discriminator = nn.Sequential(
            nn.Linear(1536,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1,bias=False)
        )
        self.project.train()
        self.discriminator.train()

        self.backbone.to(self.device)
        self.project.to(self.device)
        self.discriminator.to(self.device)

        self.th = 0.5#判别器阈值
        self.lr = 0.0002
        self.mix_noise = 1
        self.noise_std = 0.015
        self.proj_opt = torch.optim.AdamW(self.project.parameters(), self.lr*0.1)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=1e-5)

        #self.apply(self.init_weight)#初始化权重

    def forward(self, x, train=True):
        if train == False:
            self.backbone.eval()
            self.project.eval()
            self.discriminator.eval()

        #1、提取特征
        with torch.no_grad():
            self.backbone.eval()
            x = self.backbone(x)
            
        x = [self.cvt2patch(f) for f in x]#将特征图转换为patch格式，[1,1296,512,3,3] [1,324,1024,3,3]                
        
        feas = [F.adaptive_avg_pool1d(f, 1536).squeeze(1) for f in x]#特征融合 [20736, 1, 1536] [20736, 1, 1536]
        x = torch.stack(feas, dim=1)#合并 [20736, 2, 1536]

        x = x.reshape(len(x), 1, -1)#[20736,1,3072]
        x = F.adaptive_avg_pool1d(x, 1536)#[20736,1,1536]
        x = x.reshape(len(x), -1)#[20736,1536]

        self.project.train()
        x = self.project(x)#[20736, 1536]

        if train == False:
            return x

        #2、生成噪声，并添加到正样本#（优化方案：判断添加噪声的样本与未添加时的相似度，如相似度超过某一阈值则删除。）
        true_feats = x
        noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device) # (N, K)
        shape = noise_one_hot.shape
        #nn = torch.normal(0, self.noise_std, true_feats.shape)
        noise = torch.stack([torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape) for k in range(self.mix_noise)], dim=1).to(self.device) # (N, K, C)
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
        self.disc_opt.zero_grad()
        loss.backward()

        self.proj_opt.step()
        self.disc_opt.step()
        loss = loss.detach().cpu().item()

        return loss,p_true.cpu().item(),p_fake.cpu().item()

    def init_weight(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
    
    '''
    提取特征层转换为patch格式特征，并将其与第一个进行尺寸对齐
    '''
    def cvt2patch(self, feature, kerner_size=3, stride=1, pathsize=[36,36]):
        feature_shape = feature.shape
        padding = int((kerner_size - 1) / 2)
        unfolder_feature = F.unfold(feature, kernel_size=kerner_size, stride=stride, padding=padding, dilation=1) #[1, 4608, 1296] 1为batchsize,4608为一个patch的特征,1296为patch的数量
        #unfolder_feature = unfolder_feature.reshape([*unfolder_feature.shape[0:2], *feature_shape[-2:]])
        unfolder_feature = unfolder_feature.reshape([-1, *feature_shape[-2:]])
        
        feature_patch = unfolder_feature
        feature_patch = torch.unsqueeze(feature_patch, 1)
        feature_patch = F.interpolate(feature_patch, pathsize, mode="bilinear",align_corners=False)#插值为36x36尺寸
        feature_patch = torch.squeeze(feature_patch, 1)
        feature_patch = feature_patch.reshape([*feature_shape[:2],kerner_size,kerner_size,*pathsize])
        feature_patch = feature_patch.permute(0,4,5,1,2,3)
        feature_patch = feature_patch.reshape([len(feature_patch), -1, *feature_patch.shape[-3:]])
        feature_patch = feature_patch.reshape([-1,*feature_patch.shape[-3:]])

        feature_patch = feature_patch.reshape([len(feature_patch),1,-1])#20736,1,4608
        return feature_patch
    
    '''
    对齐patch
    '''
    def alignpatch(self, feature_patch, pathsize):
        feature_patch = torch.unsqueeze(feature_patch, 0)
        feature_patch = F.interpolate(feature_patch, pathsize, mode="bilinear",align_corners=False)
        feature_patch = torch.squeeze(feature_patch, 0)
        return feature_patch


    def predict(self, images):
        """Infer score and mask for a batch of images."""
        import cv2
        import numpy as np

        self.eval()
        with torch.no_grad():
            x = self(images, train=False)
            score = -self.discriminator(x)#[1296, 1]
            score = score.numpy()
            return score
    
        score = score.reshape([36,36,1])
        #print(score, torch.max(score).item(), torch.min(score).item())
        score = score.permute([2,0,1])
        score = score.unsqueeze(dim=0)
        score = F.interpolate(score, (288,288), mode="bilinear", align_corners=False)
        score = score.squeeze(dim=0)
        score = score.squeeze(dim=0)

        mask = score.detach().numpy()
        return mask

        temp = mask
        
        _,temp = cv2.threshold(temp, 0, 1, cv2.THRESH_BINARY)
        #temp = temp.astype("uint8")
        #temp = 1/(1+np.exp(-temp))#sigmoid

        max_value = np.round(np.max(mask),2)
        min_value = np.round(np.min(mask),2)
        print("\nmax=",max_value,"min=",min_value)
        temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)


        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        img = images[0]
        img = img.cpu().numpy()
        img = img.transpose([1,2,0])
        img = img*IMAGENET_STD + IMAGENET_MEAN
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        dis = cv2.hconcat([img, temp])

        cv2.imshow("dis",dis)
        cv2.waitKey()
        return
    
        torchvision.transforms.ToPILImage()(score).show()
        return
    
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features, patch_shapes = self._embed(images,provide_patch_shapes=True,evaluation=True)
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)
        return masks


if __name__ == "__main__":
    net=SimpleNet()
    x = torch.rand([16,3,288,288])
    x = net(x)

    x = torch.rand([3,1,18,18])
    y = F.interpolate(x, size=(36,36), mode="bilinear", align_corners=False)
    pass