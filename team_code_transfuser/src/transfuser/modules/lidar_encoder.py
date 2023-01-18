import torch
import torch.nn as nn
import timm


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        architecture (string): Vision architecture to be used from the TIMM model library.
        in_channels: input channels
    """

    def __init__(self, architecture, in_channels=2):
        super().__init__()

        self._model = timm.create_model(architecture, pretrained=False)
        self._model.fc = None

        if (architecture.startswith('regnet')): # Rename modules so we can use the same code
            self._model.conv1 = self._model.stem.conv
            self._model.bn1  = self._model.stem.bn
            self._model.act1 = nn.Sequential()
            self._model.maxpool =  nn.Sequential()
            self._model.layer1 = self._model.s1
            self._model.layer2 = self._model.s2
            self._model.layer3 = self._model.s3
            self._model.layer4 = self._model.s4
            self._model.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            self._model.head = nn.Sequential()

        elif (architecture.startswith('convnext')):
            self._model.conv1 = self._model.stem._modules['0']
            self._model.bn1 = self._model.stem._modules['1']
            self._model.act1 = nn.Sequential()
            self._model.maxpool = nn.Sequential()
            self._model.layer1 = self._model.stages._modules['0']
            self._model.layer2 = self._model.stages._modules['1']
            self._model.layer3 = self._model.stages._modules['2']
            self._model.layer4 = self._model.stages._modules['3']
            self._model.global_pool = self._model.head
            self._model.global_pool.flatten = nn.Sequential()
            self._model.global_pool.fc = nn.Sequential()
            self._model.head = nn.Sequential()
            _tmp = self._model.global_pool.norm
            self._model.global_pool.norm = nn.LayerNorm((self.config.perception_output_features,1,1), _tmp.eps, _tmp.elementwise_affine)

        # Change the first conv layer so that it matches the amount of channels in the LiDAR
        # Timm might be able to do this automatically
        _tmp = self._model.conv1
        use_bias = (_tmp.bias != None)
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=use_bias)
        # Need to delete the old conv_layer to avoid unused parameters
        del _tmp
        del self._model.stem.conv
        torch.cuda.empty_cache()
        if(use_bias):
            self._model.conv1.bias = _tmp.bias