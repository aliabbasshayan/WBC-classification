"""
Neural Network Architectures for Explainable WBC Classification.

Provides:
1. WBCBaselineCNN: A lightweight, custom 4-stage convolutional neural network
   designed for rapid, accurate CPU training with clean Grad-CAM feature maps.
2. WBCMobileNetV3: Transfer learning model leveraging MobileNetV3-Small with
   configurable pretrained weights and robust offline fallback.
3. Model factory and Grad-CAM target layer resolution utilities.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class WBCBaselineCNN(nn.Module):
    """
    Lightweight 4-stage Convolutional Neural Network for White Blood Cell classification.
    Parameter count: ~410k parameters.
    Optimized for fast CPU inference and high-fidelity Grad-CAM localization.
    """

    def __init__(self, num_classes: int = 5, in_channels: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        
        # Block 1: 3 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64
        )

        # Block 2: 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64 -> 32
        )

        # Block 3: 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32 -> 16
        )

        # Block 4: 128 -> 256 (Final feature extraction layer for Grad-CAM)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16 -> 8
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the last convolutional layer inside Block 4 for Grad-CAM."""
        return self.conv4[0]


class WBCMobileNetV3(nn.Module):
    """
    MobileNetV3-Small transfer learning architecture for WBC classification.
    Pretrained on ImageNet with customized classification head.
    Parameter count: ~1.8M parameters.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.2):
        super().__init__()
        
        weights = None
        if pretrained:
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
            except Exception as e:
                print(f"[Warning] Could not load default weights: {e}. Falling back to random initialization.")
                weights = None

        try:
            base_model = models.mobilenet_v3_small(weights=weights)
        except Exception as exc:
            if weights is None:
                raise
            print(
                f"[Warning] Could not load MobileNetV3-Small pretrained weights: "
                f"{exc}. Falling back to random initialization."
            )
            base_model = models.mobilenet_v3_small(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # in_features for mobilenet_v3_small is 576
        in_features = base_model.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block in features for Grad-CAM."""
        return self.features[-1]


def get_model(model_name: str = "baseline_cnn", 
              num_classes: int = 5, 
              pretrained: bool = True, 
              dropout_rate: float = 0.2) -> nn.Module:
    """
    Model factory function.

    Args:
        model_name: 'baseline_cnn' or 'mobilenet_v3_small'
        num_classes: Number of target WBC classes (default 5)
        pretrained: Whether to use pretrained weights for transfer learning
        dropout_rate: Classifier dropout rate

    Returns:
        Instantiated nn.Module.
    """
    model_name_lower = model_name.lower().replace("-", "_")
    if model_name_lower in ["baseline_cnn", "baseline", "cnn"]:
        return WBCBaselineCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name_lower in ["mobilenet_v3_small", "mobilenet", "mobilenet_v3"]:
        return WBCMobileNetV3(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Use 'baseline_cnn' or 'mobilenet_v3_small'.")
