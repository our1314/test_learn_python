"""PatchCore and PatchCore detection methods."""
import sys
sys.path.append("D:/work/program/python/DeepLearning/test_learn_python")
import logging
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cv2
import faiss
import copy
from .data import CJJDataset,DatasetSplit
from torch.utils.data import DataLoader
import torchvision
import scipy.ndimage as ndimage

# from ..patchcore import backbones
# from ..patchcore import sampler
# from ..patchcore import common

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        self.load()

    def load(
        self,
        layers_to_extract_from=('layer2', 'layer3'),
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        # featuresampler=sampler.IdentitySampler(),
        # nn_method=common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = torchvision.models.wide_resnet50_2(pretrained=True)
        self.backbone = self.backbone.to(self.device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = (3, 224, 224)
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.pretrain_embed_dimension = 1024
        self.target_embed_dimension = 1024
        self.number_of_starting_points = 10
        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = NetworkFeatureAggregator(self.backbone, self.layers_to_extract_from, self.device)
        feature_dimensions = feature_aggregator.feature_dimensions(self.input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = Preprocessing(feature_dimensions, self.pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = self.target_embed_dimension
        preadapt_aggregator = Aggregator(target_dim=self.target_embed_dimension)

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        #self.anomaly_scorer = NearestNeighbourScorer(n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method)
        self.anomaly_segmentor = RescaleSegmentor(device=self.device, target_size=self.input_shape[-2:])
        #self.featuresampler = featuresampler

        faiss.omp_set_num_threads(4)
        self.faiss_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 1024, faiss.GpuIndexFlatConfig()) #faiss.IndexFlatL2(1024)
        
    def embed(self, data):
        if data.shape[0]>1 or isinstance(data, torch.utils.data.DataLoader):#如果是DataLoader，则遍历每张图片提取特征
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)#如果是图片则直接提取特征

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):#批量转换为numpy格式
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()#改为评估模式
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)#特征聚合

        features = [features[layer] for layer in self.layers_to_extract_from]#提取layer2、layer3层特征

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]#将特征转换为patch
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:])
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(_features.unsqueeze(1),size=(ref_num_patches[0], ref_num_patches[1]),mode="bilinear",align_corners=False,)
            _features = _features.squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)#将layer2、layer3的特征融合起来
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)#将特征转换为numpy后返回

    def fit(self, training_data):#提取特征存入memory bank
        """PatchCore training.

        This function computes the embeddings of the training data and fills the memory bank of SPADE.
        """

        self._fill_memory_bank(training_data)
    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)
        #1、提取特征
        features = []
        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))#提取特征
        #2、精简特征
        features = np.concatenate(features, axis=0)#对多个张量进行拼接
        mapper = torch.nn.Linear(features.shape[1], 128, bias=False)
        features_map = mapper(torch.tensor(features))
        features_map = features_map.to(self.device)

        sample_indices = self._compute_greedy_coreset_indices(features_map)#减小特征维度，并根据论文原理减少特征数量。
        features = features[sample_indices]#根据索引进行过滤
        
        #3、添加到索引器
        self.faiss_index.add(features)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection. 运行近似迭代贪婪核集选择

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.
        这种贪婪的核心集实现不需要计算完整的N x N距离矩阵，因此需要的内存少得多，但代价是增加了采样时间。
        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(self.number_of_starting_points, None, len(features))
        start_points = np.random.choice(len(features), number_of_starting_points, replace=False).tolist()#随机找10个点
        ds = features[start_points]
        approximate_distance_matrix = self._compute_batchwise_differences(features, ds)#计算batch距离
        approximate_coreset_anchor_distances = torch.mean(approximate_distance_matrix, axis=-1).reshape(-1, 1)#求均值
        coreset_indices = []
        num_coreset_samples = int(len(features) * 0.1)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()#最大值所在索引
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(features, features[select_idx : select_idx + 1])# noqa: E203
                approximate_coreset_anchor_distances = torch.cat([approximate_coreset_anchor_distances, coreset_select_distance],dim=-1,)
                approximate_coreset_anchor_distances = torch.min(approximate_coreset_anchor_distances, dim=1).values.reshape(-1, 1)

        return np.array(coreset_indices)

    def _compute_batchwise_differences(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch. 使用pytorch计算batchwise的欧氏距离"""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)#bmm为计算两个tensor的矩阵乘法
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)
        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()#sqrt((a-b)²)

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    # labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    # masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    
    def _predict(self, images):
        """Infer score and mask for a batch of images.
            1、提取特征(一张图像尺寸为:1536x1024)
            2、faiss计算与数据库的距离(尺寸为：1536x1)
            3、
        """
        IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
        IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
        IMAGENET_MEAN = torch.reshape(IMAGENET_MEAN,[3,1,1])
        IMAGENET_STD = torch.reshape(IMAGENET_STD,[3,1,1])

        img1, img2 = images
        img1 = img1*IMAGENET_STD + IMAGENET_MEAN
        img2 = img2*IMAGENET_STD + IMAGENET_MEAN
        img1 = torch.permute(img1, [1,2,0])
        img2 = torch.permute(img2, [1,2,0])
        
        img1 = img1.numpy()
        img2 = img2.numpy()
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)


        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        score_all = []
        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)#提取特征
            features = np.asarray(features)

            patch_scores = image_scores = self.faiss_index.search(features, 1)[0]
            #patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]#采用knn进行分类，FAISS Nearest neighbourhood search.
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
            
            m1,m2 = np.array(masks[0]),np.array(masks[1])
            m3 = (m1-np.min(m1)) / (np.max(m1)-np.min(m1))
            m4 = (m2-np.min(m2)) / (np.max(m2)-np.min(m2))
            m1 = m1/5.0
            m2 = m2/5.0
            m1 = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
            m2 = cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)
            
            score_all.append(np.max(patch_scores[0]))
            score_all.append(np.max(patch_scores[1]))

            dis = cv2.vconcat([cv2.hconcat([img1,img2]), cv2.hconcat([m1,m2]), cv2.hconcat([cv2.cvtColor(m3,cv2.COLOR_GRAY2BGR), cv2.cvtColor(m4,cv2.COLOR_GRAY2BGR)])])
            print("min=", np.min(patch_scores[0]), "max=", np.max(patch_scores[0]))
            print("min=", np.min(patch_scores[1]), "max=", np.max(patch_scores[1]))
            cv2.putText(dis, str(np.max(patch_scores[0])), (0,100),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(dis, str(np.max(patch_scores[1])), (300,100),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow("dis", dis)
            cv2.waitKey()

        score_all = np.array(score_all)
        print('mean score=',np.mean(score_all), 'min score=',np.min(score_all), 'max score=',np.max(score_all))
        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(save_path, save_features_separately=False, prepend=prepend)
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules["preprocessing"].output_dim,
            "target_embed_dimension": self.forward_modules["preadapt_aggregator"].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    # def load_from_path(
    #     self,
    #     load_path: str,
    #     device: torch.device,
    #     nn_method: common.FaissNN(False, 4),
    #     prepend: str = "",
    # ) -> None:
    #     LOGGER.info("Loading and initializing PatchCore.")
    #     with open(self._params_file(load_path, prepend), "rb") as load_file:
    #         patchcore_params = pickle.load(load_file)
    #     patchcore_params["backbone"] = backbones.load(patchcore_params["backbone.name"])
    #     patchcore_params["backbone"].name = patchcore_params["backbone.name"]
    #     del patchcore_params["backbone.name"]
    #     self.load(**patchcore_params, device=device, nn_method=nn_method)

    #     self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x

class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(self.outputs, extract_layer, layers_to_extract_from[-1])
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(network_layer[-1].register_forward_hook(forward_hook))
            else:
                self.backbone.hook_handles.append(network_layer.register_forward_hook(forward_hook))
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except Exception:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]
    
class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(layer_name == last_layer_to_extract)

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise Exception()
        return None
    
class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)
    
class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)
    
class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)

class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(_scores, size=self.target_size, mode="bilinear", align_corners=False)
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [ndimage.gaussian_filter(patch_score, sigma=self.smoothing) for patch_score in patch_scores]


if __name__ == "__main__":
    patchcore = PatchCore(torch.device("cuda"))
    datasets_train = CJJDataset('D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test',split=DatasetSplit.TRAIN)
    dataloader_train = DataLoader(datasets_train, batch_size=2, shuffle=True, num_workers=8, drop_last=False)
    patchcore.fit(dataloader_train)

    datasets_test = CJJDataset('D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test',split=DatasetSplit.TEST)
    dataloader_test = DataLoader(datasets_test, batch_size=2, drop_last=True)
    patchcore.predict(dataloader_test)
    pass