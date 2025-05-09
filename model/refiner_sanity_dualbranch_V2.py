import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import pywt

# -------------------------
# BADA 模块
# -------------------------
class BlurAwareDirectionalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.att_conv = nn.Conv2d(3, 1, kernel_size=1)  # 将三个方向融合为一个 attention map
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_lh, feat_hl, feat_hh):
        # 每个方向的 L1 响应强度估计
        score_lh = torch.mean(torch.abs(feat_lh), dim=1, keepdim=True)
        score_hl = torch.mean(torch.abs(feat_hl), dim=1, keepdim=True)
        score_hh = torch.mean(torch.abs(feat_hh), dim=1, keepdim=True)

        # 拼接方向强度 → 融合 → 模糊感知 attention
        score_cat = torch.cat([score_lh, score_hl, score_hh], dim=1)
        blur_att = self.sigmoid(self.att_conv(score_cat))  # [B, 1, H, W]

        return blur_att

# -------------------------
# Encoder with BADA
# -------------------------
class WaveletEncoderV2(nn.Module):
    def __init__(self, base_channels=64, num_downsample=1, dropout_prob=0.3):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.num_downsample = num_downsample
        self.dropout_prob = dropout_prob

        self.conv_ll = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 各方向独立卷积
        self.conv_lh = nn.Conv2d(3, base_channels // 6, kernel_size=3, padding=1)
        self.conv_hl = nn.Conv2d(3, base_channels // 6, kernel_size=3, padding=1)
        self.conv_hh = nn.Conv2d(3, base_channels // 6, kernel_size=3, padding=1)

        # 模糊感知 attention
        self.bada = BlurAwareDirectionalAttention(channels=base_channels // 2)

        # 后处理
        self.hf_post = nn.Sequential(
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True)
        )

        in_channels = base_channels + base_channels // 2 + 1
        layers = []
        for _ in range(num_downsample):
            out_channels = base_channels * 2
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(self.dropout_prob))

        self.fuse = nn.Sequential(*layers)

    def forward(self, rgb, init_mask):
        Yl, Yh = self.dwt(rgb)  # Yh[0]: [B, 3, 3, H/2, W/2]
        B, C, D, H, W = Yh[0].shape

        # 分方向通道
        feat_lh = self.conv_lh(Yh[0][:, :, 0])  # [B, C', H, W]
        feat_hl = self.conv_hl(Yh[0][:, :, 1])
        feat_hh = self.conv_hh(Yh[0][:, :, 2])

        # Attention
        blur_att = self.bada(feat_lh, feat_hl, feat_hh)

        # 融合方向特征 + 应用注意力
        feat_hf = (feat_lh + feat_hl + feat_hh) * blur_att
        feat_hf = self.hf_post(feat_hf)

        # LL 分支
        feat_ll = self.conv_ll(Yl)
        mask_ds = F.interpolate(init_mask, size=feat_ll.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([feat_ll, feat_hf, mask_ds], dim=1)
        fused = self.fuse(x)

        self.cached_feats = {
            'll': Yl,
            'hf': torch.cat([feat_lh, feat_hl, feat_hh], dim=1),  # 用于 iwt
            'feat_ll': feat_ll,
            'feat_hf': feat_hf
        }

        return fused

# -------------------------
# IWT
# -------------------------
rec_hi = torch.Tensor(pywt.Wavelet('db1').rec_hi)
rec_lo = torch.Tensor(pywt.Wavelet('db1').rec_lo)
inv_filters = torch.stack([
    rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,
    rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
], dim=0)

def iwt(vres):
    B, C4, H, W = vres.shape
    C = C4 // 4
    inv_filters_exp = inv_filters.to(vres.device).unsqueeze(1)
    res = torch.zeros(B, C, H * 2, W * 2, device=vres.device)
    for i in range(C):
        x = vres[:, 4*i:4*i+4]
        x[:, 1:4] = 2 * x[:, 1:4] - 1
        temp = F.conv_transpose2d(x, inv_filters_exp, stride=2)
        res[:, i:i+1] = temp
    return res

# -------------------------
# Decoder（同前，简化展示）
# -------------------------
class WaveletDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, base_channels, dropout_prob=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.skip_gated = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, 3, padding=1),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(base_channels + skip_channels + 1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, encoder_feats, error_map):
        ll = encoder_feats['ll']
        hf = encoder_feats['hf']
        hf_split = torch.chunk(hf, 3, dim=1)
        fused = torch.cat([ll, hf_split[0], hf_split[1], hf_split[2]], dim=1)
        x = iwt(fused)

        skip_feat = torch.cat([encoder_feats['feat_ll'], encoder_feats['feat_hf']], dim=1)
        skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        error_map = F.interpolate(error_map, size=x.shape[2:], mode='bilinear', align_corners=False)

        skip_feat = self.skip_gated(skip_feat)
        x = self.proj(x)
        x = torch.cat([x, skip_feat, error_map], dim=1)
        return self.refine(x)

# -------------------------
# Final Refiner
# -------------------------
class RefinerWithDualBranch(nn.Module):
    def __init__(self, base_channels=64, num_downsample=1, dropout_prob=0.3):
        super().__init__()
        self.encoder = WaveletEncoderV2(base_channels, num_downsample, dropout_prob)
        self.skip_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.fp_decoder = WaveletDecoder(
            in_channels=3,
            skip_channels=base_channels + base_channels // 2,
            base_channels=base_channels,
            dropout_prob=dropout_prob
        )
        self.fn_decoder = WaveletDecoder(
            in_channels=3,
            skip_channels=base_channels + base_channels // 2,
            base_channels=base_channels,
            dropout_prob=dropout_prob
        )

    def forward(self, rgb, init_mask, error_map, fp_mask=None, fn_mask=None):
        enc_feat = self.encoder(rgb, init_mask)
        skip_feat = self.skip_conv(enc_feat)
        fused_feat = self.fusion_conv(torch.cat([enc_feat, skip_feat], dim=1))
        enc_feats = self.encoder.cached_feats

        correction_fp = self.fp_decoder(fused_feat, enc_feats, error_map)
        correction_fn = self.fn_decoder(fused_feat, enc_feats, error_map)

        if fp_mask is not None:
            correction_fp *= fp_mask
        if fn_mask is not None:
            correction_fn *= fn_mask

        final_pred = (init_mask - correction_fp + correction_fn).clamp(0, 1)
        return final_pred, correction_fp, correction_fn
