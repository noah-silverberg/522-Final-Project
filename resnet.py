"""
ResNet implementation as described in Deep Residual Learning for Image Recognition by He et al.
"""
# Dropout design:
# - Use Dropout2d (spatial dropout) inside the residual blocks.
# - Apply dropout after the first BN+ReLU, before the second conv.
# - No dropout in stem or classifier by default.
# - Global control with `dropout_p` passed to CifarResNet and BasicBlock.

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

class OptionAShortcut(nn.Module):
    """
    The residual connection described in the paper (Option A).
    This performs an identity mapping with striding and zero-padding of channels.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        if stride != 1 and stride != 2:
            raise ValueError("Stride must be 1 or 2.")
        
        self.stride = stride
        self.pad_channels = out_channels - in_channels

    def forward(self, x):
        # Apply striding
        x_strided = x if self.stride == 1 else x[:, :, ::2, ::2]
        
        # Apply channel padding
        if self.pad_channels > 0:
            return F.pad(x_strided, (0, 0, 0, 0, 0, self.pad_channels))
        
        return x_strided

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super().__init__()
        # We only downsample (if at all) in the first conv
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, 1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Dropout2d inside the residual block
        if dropout_p > 0.0:
            self.dropout = nn.Dropout2d(p=dropout_p)
        else:
            self.dropout = nn.Identity()

        # Shortcut: Option A if shape changes, identity otherwise
        if stride != 1 or in_channels != out_channels:
            self.shortcut = OptionAShortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Apply spatial dropout after first activation
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual addition
        out += self.shortcut(x)
        
        # Apply final ReLU after addition
        out = F.relu(out)
        return out

class CifarResNet(nn.Module):
    """
    Stage widths: 16 -> 32 -> 64. Global average pool -> linear(num_classes).
    """
    def __init__(self, n=3, num_classes=10, dropout_p=0.0):
        super().__init__()

        self.in_channels = 16
        self.dropout_p = dropout_p
        
        # Stem: conv + BN + ReLU (no dropout here by default)
        self.stem = nn.Sequential(
            conv3x3(3, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Residual layers
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)

        # Classifier: avg + flatten + FC (no dropout here by default)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride):
        """Creates a stack of n residual blocks."""
        layers = [BasicBlock(self.in_channels, out_channels,
                             stride=stride, dropout_p=self.dropout_p)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels,
                                     stride=1, dropout_p=self.dropout_p))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)

# Convenience constructors (n -> depth = 6n+2)
def resnet20(num_classes=10, dropout_p=0.0):
    return CifarResNet(n=3,  num_classes=num_classes, dropout_p=dropout_p)

def resnet32(num_classes=10, dropout_p=0.0):
    return CifarResNet(n=5,  num_classes=num_classes, dropout_p=dropout_p)

def resnet44(num_classes=10, dropout_p=0.0):
    return CifarResNet(n=7,  num_classes=num_classes, dropout_p=dropout_p)

def resnet56(num_classes=10, dropout_p=0.0):
    return CifarResNet(n=9,  num_classes=num_classes, dropout_p=dropout_p)

def resnet110(num_classes=10, dropout_p=0.0):
    return CifarResNet(n=18, num_classes=num_classes, dropout_p=dropout_p)