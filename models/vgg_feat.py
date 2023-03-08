import torch
from torchvision import models
from torch import nn


class VGGFeat(torch.nn.Module):
    """
    Input: (B, C, H, W), RGB, [-1, 1]
    """
    def __init__(self, weight_path='./pretrained_models/vgg19.pth'):
        super().__init__()
        self.model = models.vgg19(pretrained=False)
        self.build_vgg_layers()
        
        self.model.load_state_dict(torch.load(weight_path))

        self.register_parameter("RGB_mean", nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)))
        self.register_parameter("RGB_std", nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)))
        
        self.model.eval()
        # for param in self.model.parameters():
        #     param.requires_grad = True
    
    def build_vgg_layers(self):
        vgg_pretrained_features = self.model.features
        self.features = []
        feature_layers = [0, 8, 17, 26, 35]
        for i in range(len(feature_layers)-1): 
            module_layers = torch.nn.Sequential() 
            for j in range(feature_layers[i], feature_layers[i+1]):
                module_layers.add_module(str(j), vgg_pretrained_features[j])
            self.features.append(module_layers)
        self.features = torch.nn.ModuleList(self.features)

    def preprocess(self, x):
        x = (x.clone() + 1) / 2
        x = (x.clone() - self.RGB_mean) / self.RGB_std
        return x

    def forward(self, x):
        x = self.preprocess(x.clone())
        features = []
        for m in self.features:
            # print(m)
            x = m(x.clone())
            features.append(x)
        return features[::-1]
