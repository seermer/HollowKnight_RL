import torch
import numpy as np
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn import functional as F


def param_init(m, nonlinearity='relu'):  # code adapted from torchvision VGG class
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,
                                a=1e-2,
                                mode="fan_out", nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, NoisyLinear):
        m.reset_param()
        m.reset_noise()
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
        # nn.init.normal_(self.mu_W, 0, 0.01)
        # nn.init.constant_(self.mu_bias, 0)
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
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 act=nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()
        self.convs = nn.Sequential(
            act,
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            act,
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
        return x


class AbstractExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, in_channels: int,
                 activation='relu', sn=False):
        super(AbstractExtractor, self).__init__()
        self.activation_name = activation

    def forward(self, x):
        raise NotImplementedError


class ResidualExtractor(AbstractExtractor):
    def __init__(self, obs_shape: tuple, in_channels: int, activation='relu', sn=False):
        super(ResidualExtractor, self).__init__(obs_shape, in_channels, activation, sn)
        if self.activation_name == 'relu':
            act = nn.ReLU(inplace=True)
        elif self.activation_name == 'leaky_relu':
            act = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError(activation)
        out_shape = np.array(obs_shape, dtype=int)
        out_shape //= 32
        final = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        if sn:
            final = spectral_norm(final)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(48, 48, act=act),
            nn.Conv2d(48, 96, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(96, 96, act=act),
            nn.Conv2d(96, 160, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(160, 160, act=act),
            BasicBlock(160, 160, act=act),
            nn.Conv2d(160, 256, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicBlock(256, 256, act=act),
            final,
            nn.Flatten(),
        )
        self.units = 256 * np.prod(out_shape)

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class SimpleExtractor(AbstractExtractor):
    def __init__(self, obs_shape: tuple, in_channels: int, activation='relu', sn=False):
        super(SimpleExtractor, self).__init__(obs_shape, in_channels, activation, sn)
        if self.activation_name == 'relu':
            act = nn.ReLU(inplace=True)
        elif self.activation_name == 'leaky_relu':
            act = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError(activation)
        out_shape = np.array(obs_shape, dtype=int)
        out_shape //= 32
        final = nn.Conv2d(160, 320, kernel_size=3, stride=2, padding=1)
        if sn:
            final = spectral_norm(final)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(96, 160, kernel_size=3, stride=2, padding=1),
            act,
            final,
            act,
            nn.Flatten(),
        )
        self.units = 320 * np.prod(out_shape)

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class TinyExtractor(AbstractExtractor):
    def __init__(self, obs_shape: tuple, in_channels: int, activation='relu', sn=False):
        super(TinyExtractor, self).__init__(obs_shape, in_channels, activation, sn)
        if self.activation_name == 'relu':
            act = nn.ReLU(inplace=True)
        elif self.activation_name == 'leaky_relu':
            act = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError(activation)
        out_shape = np.array(obs_shape, dtype=int)
        out_shape = out_shape // 32
        final = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        if sn:
            final = spectral_norm(final)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=4),
            act,
            nn.Conv2d(32, 64, kernel_size=4, stride=4),
            act,
            final,
            act,
            nn.Flatten()
        )
        self.units = 128 * np.prod(out_shape)

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class AbstractFullyConnected(nn.Module):
    def __init__(self, extractor: AbstractExtractor, n_out: int,
                 activation='relu', noisy=False, sn=False):
        if noisy:
            assert not sn, 'spectral norm cannot be used with noisy net'
        super(AbstractFullyConnected, self).__init__()
        self.noisy = nn.ModuleList()
        self.resetable = nn.ModuleList()
        self.linear_cls = NoisyLinear if noisy else nn.Linear
        self.extractor = extractor
        if activation.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation.lower() == 'leaky_relu':
            self.act = nn.LeakyReLU(inplace=True)
        else:
            raise NotImplementedError(activation)

    def reset_noise(self):
        for layer in self.noisy:
            layer.reset_noise()

    def noise_mode(self, mode):
        for layer in self.noisy:
            layer.noise_mode(mode)

    def reset_params(self):
        n = 0
        for layer in self.resetable:
            n += 1
            param_init(layer, nonlinearity=self.extractor.activation_name)
        print(f'{n} linear layers parameter reset successfully')

    def forward(self, x, **kwargs):
        raise NotImplementedError


class SinglePathMLP(AbstractFullyConnected):
    def __init__(self, extractor: AbstractExtractor, n_out: int,
                 activation='relu', noisy=False, sn=False):
        super(SinglePathMLP, self).__init__(extractor, n_out, activation, noisy, sn)
        self.linear = self.linear_cls(extractor.units, 800)
        self.out = self.linear_cls(800, n_out)
        if sn:
            self.linear = spectral_norm(self.linear)
        if noisy:
            self.noisy.append(self.linear)
            self.noisy.append(self.out)

        self.resetable = nn.ModuleList([
            self.linear,
            self.out
        ])

        self.reset_params()

    def forward(self, x, **kwargs):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.act(x)
        x = self.out(x)
        return x


class DuelingMLP(AbstractFullyConnected):
    def __init__(self, extractor: AbstractExtractor, n_out: int,
                 activation='relu', noisy=False, sn=False):
        super(DuelingMLP, self).__init__(extractor, n_out, activation, noisy, sn)
        self.linear_val = self.linear_cls(extractor.units, 512)
        self.linear_adv = self.linear_cls(extractor.units, 512)
        self.val = self.linear_cls(512, 1)
        self.adv = self.linear_cls(512, n_out)

        if sn:
            self.linear_val = spectral_norm(self.linear_val)
            self.linear_adv = spectral_norm(self.linear_adv)

        if noisy:
            self.noisy.append(self.linear_val)
            self.noisy.append(self.linear_adv)
            self.noisy.append(self.val)
            self.noisy.append(self.adv)

        self.resetable = nn.ModuleList([
            self.linear_val,
            self.linear_adv,
            self.val,
            self.adv
        ])

        self.reset_params()

    def forward(self, x, adv_only=False, **kwargs):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        adv = self.linear_adv(x)
        adv = self.act(adv)
        adv = self.adv(adv)
        if adv_only:
            return adv
        val = self.linear_val(x)
        val = self.act(val)
        val = self.val(val)
        x = val + adv - adv.mean(dim=1, keepdim=True)
        return x
