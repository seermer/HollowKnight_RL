import torch
import torchvision
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.ops.misc import SqueezeExcitation as SE


def param_init(m):  # code adapted from torchvision VGG class
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


class NoisyLinear(nn.Module):
    """
    NoisyLinear code adapted from
    https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch/blob/master/fqf_iqn_qrdqn/network.py
    """

    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self._noise_mode = True

        self.reset_param()
        self.reset_noise()

    @staticmethod
    def _f(x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def reset_param(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def reset_noise(self):
        self.eps_p.copy_(self._f(self.eps_p))
        self.eps_q.copy_(self._f(self.eps_q))

    def noise_mode(self, mode):
        self._noise_mode = mode

    def forward(self, x):
        if self._noise_mode:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)


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
    def __init__(self, obs_shape: tuple, n_frames: int):
        super(VGGExtractor, self).__init__()
        self.convs = torchvision.models.vgg11().features[:-1]
        self.convs[0] = nn.Conv2d(n_frames, 64, 5, stride=2, padding=2)
        self.out_shape = np.array((512,) + tuple(obs_shape), dtype=int)
        self.out_shape[1:] //= 32

        param_init(self.convs[0])

    def forward(self, x):
        x = self.convs(x)
        return x


class ResidualExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, n_frames: int):
        super(ResidualExtractor, self).__init__()
        self.out_shape = np.array((1024,) + tuple(obs_shape), dtype=int)
        self.out_shape[1:] //= 32
        act = nn.ReLU(inplace=True)
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 48, 4, 4),
            act,
            BasicBlock(48, 48),
            BasicBlock(48, 96, 2),
            BasicBlock(96, 160, 2),
            BasicBlock(160, 160),
            BasicBlock(160, 256, 2),
            BasicBlock(256, 256),
            nn.Conv2d(256, 1024, 1),
            act,
            nn.AvgPool2d(tuple(self.out_shape[1:]))
        )
        self.out_shape[1:] = [1, 1]

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class SimpleExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, n_frames: int):
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
        )
        self.out_shape = np.array((384,) + tuple(obs_shape), dtype=int)
        self.out_shape[1:] //= 32

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class AttentionExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, n_frames: int):
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

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class AbstractFullyConnected(nn.Module):
    def __init__(self, extractor: nn.Module, n_out: int, pool=True, noisy=False):
        super(AbstractFullyConnected, self).__init__()
        self.noisy = nn.ModuleList()
        self.linear_cls = NoisyLinear if noisy else nn.Linear
        self.extractor = extractor
        self.pool = nn.AvgPool2d(tuple(extractor.out_shape[1:])) if pool else lambda x: x
        self.units = extractor.out_shape[0]
        if not pool:
            self.units *= int(np.prod(extractor.out_shape[1:]))

        self.act = nn.ReLU(inplace=True)

    def reset_noise(self):
        for layer in self.noisy:
            layer.reset_noise()

    def noise_mode(self, mode):
        for layer in self.noisy:
            layer.noise_mode(mode)

    def forward(self, x):
        raise NotImplementedError


class SinglePathMLP(AbstractFullyConnected):
    def __init__(self, extractor: nn.Module, n_out: int, pool=True, noisy=False):
        super(SinglePathMLP, self).__init__(extractor, n_out, pool, noisy)
        self.linear = self.linear_cls(self.units, 512)
        self.out = self.linear_cls(512, n_out)
        if noisy:
            self.noisy.append(self.linear)
            self.noisy.append(self.out)

        param_init(self.linear)
        param_init(self.out)

    def forward(self, x):
        x = self.extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.act(x)
        x = self.out(x)
        return x


class DuelingMLP(AbstractFullyConnected):
    def __init__(self, extractor: nn.Module, n_out: int, pool=True, noisy=False):
        super(DuelingMLP, self).__init__(extractor, n_out, pool, noisy)
        self.linear_val = self.linear_cls(self.units, 320)
        self.linear_adv = self.linear_cls(self.units, 320)
        self.val = self.linear_cls(320, 1)
        self.adv = self.linear_cls(320, n_out)

        if noisy:
            self.noisy.append(self.linear_val)
            self.noisy.append(self.linear_adv)
            self.noisy.append(self.val)
            self.noisy.append(self.adv)

        param_init(self.linear_adv)
        param_init(self.linear_val)
        param_init(self.adv)
        param_init(self.val)

    def forward(self, x):
        x = self.extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        adv = self.linear_adv(x)
        adv = self.act(adv)
        adv = self.adv(adv)
        val = self.linear_val(x)
        val = self.act(val)
        val = self.val(val)
        x = val + adv - adv.mean(dim=1, keepdim=True)
        return x
