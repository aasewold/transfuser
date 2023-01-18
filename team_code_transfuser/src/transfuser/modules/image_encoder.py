import torch.nn as nn
import timm
      

class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, architecture, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = timm.create_model(architecture, pretrained=True)
        self.features.fc = None
        # Delete parts of the networks we don't want
        if (architecture.startswith('regnet')): # Rename modules so we can use the same code
            self.features.conv1 = self.features.stem.conv
            self.features.bn1  = self.features.stem.bn
            self.features.act1 = nn.Sequential() #The Relu is part of the batch norm here.
            self.features.maxpool =  nn.Sequential()
            self.features.layer1 =self.features.s1
            self.features.layer2 =self.features.s2
            self.features.layer3 =self.features.s3
            self.features.layer4 =self.features.s4
            self.features.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self.features.head = nn.Sequential()

        elif (architecture.startswith('convnext')):
            self.features.conv1 = self.features.stem._modules['0']
            self.features.bn1 = self.features.stem._modules['1']
            self.features.act1 = nn.Sequential()  # Don't see any activatin function after the stem. Need to verify
            self.features.maxpool = nn.Sequential()
            self.features.layer1 = self.features.stages._modules['0']
            self.features.layer2 = self.features.stages._modules['1']
            self.features.layer3 = self.features.stages._modules['2']
            self.features.layer4 = self.features.stages._modules['3']
            self.features.global_pool = self.features.head
            self.features.global_pool.flatten = nn.Sequential()
            self.features.global_pool.fc = nn.Sequential()
            self.features.head = nn.Sequential()
            # ConvNext don't have the 0th entry that res nets use.
            self.features.feature_info.append(self.features.feature_info[3])
            self.features.feature_info[3] = self.features.feature_info[2]
            self.features.feature_info[2] = self.features.feature_info[1]
            self.features.feature_info[1] = self.features.feature_info[0]

            #This layer norm is not pretrained anymore but that shouldn't matter since it is the last layer in the network.
            _tmp = self.features.global_pool.norm
            self.features.global_pool.norm = nn.LayerNorm((self.config.perception_output_features,1,1), _tmp.eps, _tmp.elementwise_affine)
