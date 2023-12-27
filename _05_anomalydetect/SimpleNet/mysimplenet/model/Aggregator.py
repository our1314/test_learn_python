import torch
import torch.nn.functional as F


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
    

if __name__ == "__main__":
    net = Aggregator(1536)
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(total_trainable_params)
    pass