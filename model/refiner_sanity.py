import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward  # pip install git+https://github.com/fbcotter/pytorch_wavelets


class WaveletEncoderV2(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')

        # LL 分支（主干）
        self.conv_ll = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # HF 分支（细节）
        self.conv_hf = nn.Sequential(
            nn.Conv2d(9, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 融合 + 下采样（与 init mask 一起）
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels + base_channels // 2 + 1, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, init_mask):
        Yl, Yh = self.dwt(rgb)  # Yl: LL, Yh[0]: high-frequency bands
        ll = Yl

        # 处理 Yh[0] 为 [B, 9, H/2, W/2] 格式
        if Yh[0].dim() == 5:
            # shape = [B, D=3, C=3, H/2, W/2] → permute & reshape to [B, 9, H/2, W/2]
            hf = Yh[0].permute(0, 2, 1, 3, 4).reshape(Yh[0].shape[0], -1, Yh[0].shape[3], Yh[0].shape[4])
        else:
            # fallback if shape is already [B, 9, H/2, W/2]
            hf = Yh[0]

        feat_ll = self.conv_ll(ll)
        feat_hf = self.conv_hf(hf)

        mask_ds = F.interpolate(init_mask, size=feat_ll.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([feat_ll, feat_hf, mask_ds], dim=1)
        fused = self.fuse(x)
        return fused


class RefinerWithWaveletEncoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()

        self.encoder = WaveletEncoderV2(base_channels=base_channels)

        # Skip & Fusion（仍使用残差增强）
        self.skip_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder（与原模型兼容 + 补全上采样）
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 2 + 1, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # ✅ 新增上采样层

    def forward(self, rgb, init_mask, error_map):
        enc_feat = self.encoder(rgb, init_mask)  # 输出为 base_channels*2
        skip_feat = self.skip_conv(enc_feat)
        fused_feat = self.fusion_conv(torch.cat([enc_feat, skip_feat], dim=1))

        error_map_ds = F.interpolate(error_map, size=fused_feat.shape[2:], mode='bilinear', align_corners=False)
        decoder_input = torch.cat([fused_feat, error_map_ds], dim=1)
        m_pred = self.decoder(decoder_input)
        m_pred = self.upsample(m_pred)  # ✅ 补回原始输入尺寸
        return m_pred
