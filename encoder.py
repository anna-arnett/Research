from importlib.util import module_for_loader
from dataloader import *
import torch.nn as nn
import torchvision.models as models
from dataloader import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.base_model = models.densenet161(pretrained=True).features
        #self.base_model = models.densenet161(weights='DEFAULT').features

        self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
        self.feat_out_channels = [96, 96, 192, 384, 2208]


    def forward(self, x):
        
        feats = []
        for name, module in self.base_model._modules.items():
            x = module(x) #foward pass through each module. 
            if any(x in name for x in self.feat_names): #if the name of the module is in the list of feature names, append it to the list of features
                #print("appending result of ", name)
                feats.append(x)

        return feats


if __name__ == '__main__':

    nyu_data = NYUDatasetRGBD(TRAIN_PATHS)
    sample = nyu_data.__getitem__(0)['image']
    sample = sample.unsqueeze(0) #add batch dimension
    print(f'Image Type: {type(sample)}')
    print(f'Image Shape: {sample.shape}')
    densenet_161 = Encoder()
    result = densenet_161.forward(sample)
    print(f'Output Length: {len(result)}')
    for i in range(len(result)):
        print(f'Feature {i} Shape: {result[i].shape}')