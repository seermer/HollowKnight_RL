import torch
from torch import nn


class Intensity(nn.Module):
    # code adapted from paper
    # Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels
    # https://arxiv.org/abs/2004.13649
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise
