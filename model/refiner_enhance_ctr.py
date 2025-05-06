import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x1, x2, x3


class BasicDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(64)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(32)
        )
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x3):
        x = self.up1(x3)
        x = self.up2(x)
        return self.out(x)


class CTRGuidanceModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feat, m1, gt_alpha):
        # Soft region where CTR is meaningful
        soft_region = ((gt_alpha > 0.05) & (gt_alpha < 0.95) &
                       (m1 > 0.05) & (m1 < 0.95)).float()
        diff = torch.abs(m1 - gt_alpha) * soft_region
        pooled = self.global_pool(diff)  # shape: (B, 1, 1, 1)

        # Broadcast attention and apply
        pooled = pooled.view(pooled.size(0), -1)  # flatten
        weight = self.fc(pooled).view(feat.size(0), feat.size(1), 1, 1)
        return feat * weight


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BasicEncoder()
        self.decoder = BasicDecoder()
        self.ctr_module = CTRGuidanceModule(in_channels=128)

    def forward(self, image, m1, gt_alpha=None):
        # Input concat: RGB + coarse mask
        x = torch.cat([image, m1], dim=1)

        # Encode
        x1, x2, x3 = self.encoder(x)

        # Inject CTR-based guidance at x3 (stage 3)
        if self.training and gt_alpha is not None:
            x3 = self.ctr_module(x3, m1, gt_alpha)

        # Decode
        out = self.decoder(x3)
        return out
