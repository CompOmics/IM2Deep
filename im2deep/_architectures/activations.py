"""Activation functions for IM2Deep models."""

import torch
from torch import nn


class LRelu_with_saturation(nn.Module):
    def __init__(self, negative_slope, saturation):
        super().__init__()
        self.negative_slope = negative_slope
        self.saturation = saturation
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)

    def forward(self, x):
        activated = self.leaky_relu(x)
        return torch.clamp(activated, max=self.saturation)
