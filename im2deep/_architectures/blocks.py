"""Building blocks for the IM2Deep architecture."""

from torch import nn

from im2deep._architectures.activations import LRelu_with_saturation


class Conv1dActivation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        initializer,
        negative_slope,
        saturation,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.initializer = initializer
        self.activation = LRelu_with_saturation(
            negative_slope=negative_slope, saturation=saturation
        )

        initializer(self.conv.weight, 0.0, 0.05)

    def forward(self, x):
        return self.activation(self.conv(x))


class DenseActivation(nn.Module):
    def __init__(self, in_features, out_features, initializer, negative_slope, saturation):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.initializer = initializer
        self.activation = LRelu_with_saturation(
            negative_slope=negative_slope, saturation=saturation
        )

        initializer(self.linear.weight, 0.0, 0.05)

    def forward(self, x):
        return self.activation(self.linear(x))
