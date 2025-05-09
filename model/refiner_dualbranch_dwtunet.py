import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pywt

# 定义 db1 小波滤波器
w = pywt.Wavelet('db1')
dec_hi = torch.Tensor(w.dec_hi[::-1])
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

# 构造正向和反向卷积核
filters = torch.stack([
    dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1) / 2.0,
    dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
    dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
    dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
], dim=0)

inv_filters = torch.stack([
    rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,
    rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
], dim=0)

# 正向 DWT
def wt(vimg):
    B, C, H, W = vimg.shape
    filters_exp = filters.to(vimg.device).unsqueeze(1)  # [4, 1, k, k]
    res = torch.zeros(B, 4 * C, H // 2, W // 2, device=vimg.device)

    for i in range(C):
        v = vimg[:, i:i+1]
        r = F.conv2d(v, filters_exp, stride=2)
        r[:, 1:4] = (r[:, 1:4] + 1) / 2.0
        res[:, 4*i:4*i+4] = r
    return res

# 逆向 IWT
def iwt(vres):
    B, C4, H, W = vres.shape
    C = C4 // 4
    inv_filters_exp = inv_filters.to(vres.device).unsqueeze(1)
    res = torch.zeros(B, C, H * 2, W * 2, device=vres.device)

    for i in range(C):
        x = vres[:, 4*i:4*i+4]
        x[:, 1:4] = 2 * x[:, 1:4] - 1
        r = F.conv_transpose2d(x, inv_filters_exp, stride=2)
        res[:, i:i+1] = r
    return res


class WaveletEncoderV3(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_levels=3, dropout_prob=0.3):
        super().__init__()
        self.num_levels = num_levels
        self.dropout_prob = dropout_prob

        self.enc_blocks = nn.ModuleList()
        self.hf_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()

        current_channels = in_channels
        for level in range(num_levels):
            hf_conv = nn.Sequential(
                nn.Conv2d(3 * current_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
            self.hf_convs.append(hf_conv)

            block = nn.Sequential(
                nn.Conv2d(base_channels + current_channels, base_channels * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.dropout_prob)
            )
            self.enc_blocks.append(block)

            mask_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
            self.mask_convs.append(mask_conv)

            current_channels = base_channels * 2
            base_channels *= 2

    def forward(self, x, init_mask):
        feat = x
        mask = init_mask

        for i in range(self.num_levels):
            x_wt = wt(feat)
            B, C4, H, W = x_wt.shape
            C = C4 // 4

            LL = x_wt[:, 0::4]
            HF = torch.cat([x_wt[:, 1::4], x_wt[:, 2::4], x_wt[:, 3::4]], dim=1)

            hf_feat = self.hf_convs[i](HF)

            mask = F.interpolate(mask, size=LL.shape[2:], mode='bilinear', align_corners=False)
            mask_feat = self.mask_convs[i](mask)

            feat = torch.cat([LL, hf_feat], dim=1)
            feat = self.enc_blocks[i](feat)

        return feat


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, base_channels, num_upsample, dropout_prob=0.3):
        super().__init__()
        layers = []
        current_channels = in_channels

        for i in range(num_upsample):
            next_channels = current_channels // 2
            layers.append(nn.ConvTranspose2d(current_channels, next_channels, 4, stride=2, padding=1))
            if i == num_upsample - 1:
                layers.append(nn.BatchNorm2d(next_channels))
            layers.append(nn.ReLU(inplace=True))
            current_channels = next_channels

        self.upsample_layers = nn.Sequential(*layers)
        self.final_channels = current_channels

        self.fuse_and_predict = nn.Sequential(
            nn.Conv2d(self.final_channels + 1, self.final_channels, 3, padding=1),
            nn.BatchNorm2d(self.final_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(self.final_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, error_map_fullres):
        x = self.upsample_layers(x)
        assert error_map_fullres.shape[2:] == x.shape[2:], \
            f"Mismatch after upsampling: decoder={x.shape}, error_map={error_map_fullres.shape}"
        x = torch.cat([x, error_map_fullres], dim=1)
        return self.fuse_and_predict(x)


class RefinerWithDualBranch(nn.Module):
    def __init__(self, base_channels=64, num_downsample=3, dropout_prob=0.3):
        super().__init__()
        self.encoder = WaveletEncoderV3(in_channels=3, base_channels=base_channels, num_levels=num_downsample, dropout_prob=dropout_prob)

        mid_channels = base_channels * (2 ** num_downsample)
        self.skip_conv = nn.Conv2d(mid_channels, mid_channels, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.fp_decoder = DecoderBlock(mid_channels, base_channels, num_upsample=num_downsample, dropout_prob=dropout_prob)
        self.fn_decoder = DecoderBlock(mid_channels, base_channels, num_upsample=num_downsample, dropout_prob=dropout_prob)

    def forward(self, rgb, init_mask, error_map, fp_mask=None, fn_mask=None):
        enc_feat = self.encoder(rgb, init_mask)
        skip_feat = self.skip_conv(enc_feat)
        fused_feat = self.fusion_conv(torch.cat([enc_feat, skip_feat], dim=1))

        correction_fp = self.fp_decoder(fused_feat, error_map)
        correction_fn = self.fn_decoder(fused_feat, error_map)

        if fp_mask is not None:
            correction_fp = correction_fp * fp_mask
        if fn_mask is not None:
            correction_fn = correction_fn * fn_mask

        final_pred = (init_mask - correction_fp + correction_fn).clamp(0, 1)
        return final_pred, correction_fp, correction_fn
