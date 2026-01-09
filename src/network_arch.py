# ==========================================
# Network Architecture: WRN + GroupNorm + WS
# ==========================================
import torch.nn as nn
import torch.nn.functional as F

class WSConv2d(nn.Conv2d):
    """Weight Standardized Convolution"""
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.GroupNorm(16, in_planes)
        self.conv1 = WSConv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(16, planes)
        self.conv2 = WSConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                WSConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=10):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        n = (depth - 4) // 6
        self.conv1 = WSConv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(nChannels[0], nChannels[1], n, stride=1)
        self.layer2 = self._make_layer(nChannels[1], nChannels[2], n, stride=2)
        self.layer3 = self._make_layer(nChannels[2], nChannels[3], n, stride=2)
        self.bn1 = nn.GroupNorm(16, nChannels[3])
        self.linear = nn.Linear(nChannels[3], num_classes)

    def _make_layer(self, in_planes, out_planes, nb_layers, stride):
        layers = [WideBasic(in_planes, out_planes, stride)]
        for _ in range(1, nb_layers):
            layers.append(WideBasic(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out