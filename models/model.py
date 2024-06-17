import torch.nn as nn
from models.functions import ReverseLayerF
import torch
from torchvision import models
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'lse']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'lse'], no_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        
        pre_a = models.alexnet(pretrained=True)
        self.features = pre_a.features[0:8]
        for param in self.features.parameters():
            param.requires_grad = False

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)

        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(64)

        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same', dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.bottleneck4 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same', dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 100),  # Adjusted input size
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 100),  # Adjusted input size
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, dilation=2)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, dilation=2)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, input_data, alpha):
        feature = self.features(input_data)
        
        feature = self.relu(self.bn1(self.conv4(feature)))
        feature = self.bottleneck2(feature)
        feature = self.cbam1(feature)
        feature = self.relu(self.bn2(self.conv5(feature)))
        feature = self.bottleneck4(feature)
        feature = self.cbam2(feature)
        feature = self.max3(feature)
        
        # Print the shape of the tensor before view operation
        print("Shape before view:", feature.shape)
        
        feature = feature.view(feature.size(0), -1)  # Flatten the tensor appropriately
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output