from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBnPReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )


class ConvBn(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBnPReLU(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBnPReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.layers = nn.Sequential(
            ConvBnPReLU(
                in_channels,
                hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            ConvBnPReLU(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels,
            ),
            ConvBn(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.activation = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        if self.use_residual:
            out = out + x
        return self.activation(out)


class TinyLiveness(nn.Module):
    """Tiny face liveness classifier for CPU/mobile deployment.

    Input is a normalized RGB face tensor shaped ``(N, 3, 112, 112)``.
    Output is a raw live logit shaped ``(N, 1)``. Apply sigmoid to get the
    probability that the crop is a real live face.
    """

    def __init__(self, width_mult: float = 0.5, dropout: float = 0.1) -> None:
        super().__init__()
        c16 = self._make_divisible(16 * width_mult)
        c32 = self._make_divisible(32 * width_mult)
        c64 = self._make_divisible(64 * width_mult)
        c96 = self._make_divisible(96 * width_mult)
        c128 = self._make_divisible(128 * width_mult)

        self.features = nn.Sequential(
            ConvBnPReLU(3, c16, kernel_size=3, stride=2),  # 112 -> 56
            DepthwiseSeparableBlock(c16, c32, stride=2),  # 56 -> 28
            ResidualBottleneckBlock(c32, c32, stride=1, expansion=2),
            ResidualBottleneckBlock(c32, c64, stride=2, expansion=2),  # 28 -> 14
            ResidualBottleneckBlock(c64, c64, stride=1, expansion=2),
            ResidualBottleneckBlock(c64, c96, stride=2, expansion=2),  # 14 -> 7
            ResidualBottleneckBlock(c96, c96, stride=1, expansion=2),
            ConvBnPReLU(c96, c128, kernel_size=1, stride=1, padding=0),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c128, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _make_divisible(value: float, divisor: int = 8) -> int:
        return max(divisor, int(value + divisor / 2) // divisor * divisor)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d | nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


class TinyLivenessProbability(nn.Module):
    """Deployment wrapper that returns live probability instead of raw logits."""

    def __init__(self, model: TinyLiveness) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.model(x))


class MobileNetV3Liveness(nn.Module):
    """MobileNetV3 liveness classifier.

    This is still mobile/CPU friendly, but has ImageNet pretraining available
    for stronger transfer when the liveness dataset is small or shifted.
    """

    def __init__(
        self,
        *,
        dropout: float = 0.2,
        pretrained: bool = False,
        width_mult: float = 1.0,
        variant: str = "small",
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import (
                MobileNet_V3_Large_Weights,
                MobileNet_V3_Small_Weights,
                mobilenet_v3_large,
                mobilenet_v3_small,
            )
        except ImportError as exc:
            raise ImportError(
                "MobileNetV3Liveness requires torchvision. Install the training requirements."
            ) from exc

        if pretrained and width_mult != 1.0:
            raise ValueError("pretrained MobileNetV3-Small only supports width_mult=1.0")

        variant = variant.strip().lower()
        if variant == "large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_large(weights=weights, width_mult=width_mult, dropout=dropout)
        elif variant == "small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights, width_mult=width_mult, dropout=dropout)
        else:
            raise ValueError(f"unknown MobileNetV3 variant: {variant!r}")
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, 1)
        nn.init.xavier_normal_(self.backbone.classifier[-1].weight)
        nn.init.zeros_(self.backbone.classifier[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


class TorchvisionClassifierLiveness(nn.Module):
    """Small torchvision classifier adapted to a single live logit."""

    def __init__(self, architecture: str, *, dropout: float = 0.2, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError(
                "TorchvisionClassifierLiveness requires torchvision. Install the training requirements."
            ) from exc

        architecture = architecture.strip().lower()
        if architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights, dropout=dropout)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, 1)
        elif architecture == "shufflenet_v2_x1_0":
            weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
            self.backbone = models.shufflenet_v2_x1_0(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 1)
        elif architecture == "shufflenet_v2_x1_5":
            weights = models.ShuffleNet_V2_X1_5_Weights.DEFAULT if pretrained else None
            self.backbone = models.shufflenet_v2_x1_5(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 1)
        elif architecture == "shufflenet_v2_x2_0":
            weights = models.ShuffleNet_V2_X2_0_Weights.DEFAULT if pretrained else None
            self.backbone = models.shufflenet_v2_x2_0(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"unsupported torchvision liveness architecture: {architecture!r}")

        self._init_last_layer()

    def _init_last_layer(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.out_features == 1:
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


def build_liveness_model(
    architecture: str = "tiny",
    *,
    width_mult: float = 0.5,
    dropout: float = 0.1,
    pretrained: bool = False,
) -> nn.Module:
    normalized = architecture.strip().lower()
    if normalized in {"tiny", "tinyliveness"}:
        if pretrained:
            raise ValueError("TinyLiveness scratch CNN does not support pretrained weights")
        return TinyLiveness(width_mult=width_mult, dropout=dropout)
    if normalized in {"mobilenet_v3_small", "mobilenetv3_small", "mobilenet"}:
        return MobileNetV3Liveness(
            width_mult=width_mult,
            dropout=dropout,
            pretrained=pretrained,
            variant="small",
        )
    if normalized in {"mobilenet_v3_large", "mobilenetv3_large"}:
        return MobileNetV3Liveness(
            width_mult=width_mult,
            dropout=dropout,
            pretrained=pretrained,
            variant="large",
        )
    if normalized in {"efficientnet_b0", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"}:
        if width_mult != 1.0:
            raise ValueError(f"{architecture} does not support width_mult")
        return TorchvisionClassifierLiveness(
            normalized,
            dropout=dropout,
            pretrained=pretrained,
        )
    raise ValueError(f"unknown liveness architecture: {architecture!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
