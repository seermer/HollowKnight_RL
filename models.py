import torch
import torchvision
import numpy as np
from torch import nn
from torchvision.ops.misc import SqueezeExcitation as SE


def param_init(m):  # code adapted from torchvision VGG class
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            self.act,
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(stride, stride),
            nn.Conv2d(in_channels, out_channels, 1)
        ) if stride > 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        feat_map = self.convs(x)
        shortcut = self.shortcut(x)
        x = feat_map + shortcut
        x = self.act(x)
        return x


class VGGExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, n_frames: int, device=None):
        super(VGGExtractor, self).__init__()
        self.convs = torchvision.models.vgg11().features[:-1]
        self.convs[0] = nn.Conv2d(n_frames, 64, 5, stride=2, padding=2)
        self.out_shape = np.array((512,) + tuple(obs_shape), dtype=int)
        self.out_shape[1:] //= 32
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        param_init(self.convs[0])

        self.to(self.device)

    def forward(self, x):
        x = self.convs(x)
        return x


class ResidualExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, n_frames: int, device=None):
        super(ResidualExtractor, self).__init__()
        self.out_shape = np.array((800,) + tuple(obs_shape), dtype=int)
        self.out_shape[1:] //= 32
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 48, 4, 4),
            BasicBlock(48, 48),
            BasicBlock(48, 96, 2),
            BasicBlock(96, 160, 2),
            BasicBlock(160, 160),
            BasicBlock(160, 320, 2),
            BasicBlock(320, 320),
            nn.Conv2d(320, 800, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(tuple(self.out_shape[1:]))
        )
        self.out_shape[1:] = [1, 1]
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        for m in self.modules():
            param_init(m)

        self.to(self.device)

    def forward(self, x):
        x = self.convs(x)
        return x


class SimpleExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, n_frames: int, device=None):
        super(SimpleExtractor, self).__init__()
        act = nn.ReLU(inplace=True)
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 64, kernel_size=7, stride=4, padding=3),
            act,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            act,
        )
        self.out_shape = np.array((384,) + tuple(obs_shape), dtype=int)
        self.out_shape[1:] //= 32
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        for m in self.modules():
            param_init(m)

        self.to(self.device)

    def forward(self, x):
        x = self.convs(x)
        return x


class AttentionExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, n_frames: int, device=None):
        super(AttentionExtractor, self).__init__()
        act = nn.ReLU(inplace=True)
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            act,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            act,
            SE(128, 8, activation=lambda: act),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            act,
            SE(256, 16, activation=lambda: act),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            act,
            SE(384, 24, activation=lambda: act),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            act,
        )
        self.out_shape = np.array((384,) + tuple(obs_shape), dtype=int)
        self.out_shape[1:] //= 32
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        for m in self.modules():
            param_init(m)

        self.to(self.device)

    def forward(self, x):
        x = self.convs(x)
        return x


class SinglePathMLP(nn.Module):
    def __init__(self, extractor: nn.Module, n_out: int, pool=True):
        super(SinglePathMLP, self).__init__()
        self.extractor = extractor
        self.pool = nn.AvgPool2d(tuple(extractor.out_shape[1:])) if pool else nn.Identity()
        units = extractor.out_shape[0]
        if not pool:
            units *= int(np.prod(extractor.out_shape[1:]))
        self.linear = nn.Linear(units, 512)
        self.out = nn.Linear(512, n_out)
        self.act = nn.ReLU(inplace=True)
        self.device = extractor.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        param_init(self.linear)
        param_init(self.out)

        self.to(self.device)

    def forward(self, x):
        x = self.extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.act(x)
        x = self.out(x)
        return x


class DuelingMLP(nn.Module):
    def __init__(self, extractor: nn.Module, n_out: int, pool=True):
        super(DuelingMLP, self).__init__()
        self.extractor = extractor
        self.pool = nn.AvgPool2d(tuple(extractor.out_shape[1:])) if pool else nn.Identity()
        units = extractor.out_shape[0]
        if not pool:
            units *= int(np.prod(extractor.out_shape[1:]))
        self.linear_val = nn.Linear(units, 320)
        self.linear_adv = nn.Linear(units, 320)
        self.val = nn.Linear(320, 1)
        self.adv = nn.Linear(320, n_out)
        self.act = nn.ReLU(inplace=True)
        self.device = extractor.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        param_init(self.linear_adv)
        param_init(self.linear_val)
        param_init(self.adv)
        param_init(self.val)

        self.to(self.device)

    def forward(self, x):
        x = self.extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        val = self.linear_val(x)
        val = self.act(val)
        adv = self.linear_adv(x)
        adv = self.act(adv)
        val = self.val(val)
        adv = self.adv(adv)
        x = val + adv - adv.mean(dim=1, keepdim=True)
        return x
