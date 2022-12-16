from torch import nn
import torch
import torchvision
import numpy as np


def param_init(m):  # code adapted from torchvision VGG class
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


class VGGExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, n_frames: int, device=None):
        super(VGGExtractor, self).__init__()
        self.convs = torchvision.models.vgg11().features[:-1]
        self.convs[0] = nn.Conv2d(n_frames, 64, 5, stride=2, padding=2)
        self.out_shape = np.array((512,) + tuple(obs_shape), dtype=int)
        self.out_shape[1:] //= 32
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
        act = nn.ReLU(True)
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 48, kernel_size=7, stride=4, padding=3),
            act,
            nn.Conv2d(48, 80, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(80, 128, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            act,
        )
        self.out_shape = np.array((256,) + tuple(obs_shape), dtype=int)
        self.out_shape[1:] //= 32
        self.out_shape[1:] -= 2
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
        self.linear1 = nn.Linear(units, 512)
        self.linear2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, n_out)
        self.act = nn.ReLU(True)
        self.device = extractor.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        for m in self.modules():
            param_init(m)

        self.to(self.device)

    def forward(self, x):
        x = self.extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.out(x)
        return x


class DuelingMLP(nn.Module):
    def __init__(self, extractor: nn.Module, action_dims: list):
        raise NotImplementedError
        super(DuelingMLP, self).__init__()
        self.extractor = extractor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(extractor.out_shape[0], 1024)
        self.branches = nn.ModuleList([nn.Linear(1024, 256)
                                       for _ in range(len(action_dims))])
        self.outs = nn.ModuleList([nn.Linear(256, n_actions)
                                   for n_actions in action_dims])
        self.act = nn.ReLU(True)

        for m in self.modules():
            param_init(m)

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError
        x = self.extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        xs = []
        for linear2, clf in zip(self.branches, self.outs):
            branch = linear2(x)
            branch = self.act(branch)
            branch = clf(branch)
            xs.append(branch)
        x = torch.hstack(xs)
        return x
