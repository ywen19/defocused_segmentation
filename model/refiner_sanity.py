import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinerWithErrorMap(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(4, base_channels, 3, padding=1),  # RGB+Init
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # ---------- Skip & Fusion ----------
        self.skip_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # ---------- Decoder w/ Error Map ----------
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 2 + 1, base_channels, 3, padding=1),  # +1 for error_map
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, init_mask, error_map):
        x = torch.cat([rgb, init_mask], dim=1)
        enc_feat = self.encoder(x)
        skip_feat = self.skip_conv(enc_feat)
        fused_feat = self.fusion_conv(torch.cat([enc_feat, skip_feat], dim=1))

        # ✅ 结构引导图尺寸适配
        error_map_ds = F.interpolate(error_map, size=fused_feat.shape[2:], mode='bilinear', align_corners=False)
        decoder_input = torch.cat([fused_feat, error_map_ds], dim=1)

        m_pred = self.decoder(decoder_input)
        return m_pred
