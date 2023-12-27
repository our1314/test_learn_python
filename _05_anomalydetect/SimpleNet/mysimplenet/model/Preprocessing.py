import torch
import torch.nn.functional as F


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)#[20736, 1, 4608] [20736, 1, 9216] 
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)

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
            _features.append(module(feature))#[20736, 1536]
        return torch.stack(_features, dim=1)
    
if __name__ == "__main__":
    net = Preprocessing([1536,1536],[1536,1536])
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(total_trainable_params)