import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import pywt
from torch.autograd import Variable

w = pywt.Wavelet('db1')
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

inv_filters = torch.stack([
    rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,
    rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
], dim=0)  # shape: [4, k, k]

def iwt(vres):
    """
    Inverse wavelet transform for input [B, 4*C, H, W]
    Reconstructs [B, C, H*2, W*2]
    """
    B, C4, H, W = vres.shape
    C = C4 // 4
    inv_filters_exp = inv_filters.to(vres.device).unsqueeze(1)  # [4, 1, k, k]

    res = torch.zeros(B, C, H * 2, W * 2, device=vres.device)

    for i in range(C):
        x = vres[:, 4*i:4*i+4]  # [B, 4, H, W]
        x[:, 1:4] = 2 * x[:, 1:4] - 1  # reverse shift
        temp = F.conv_transpose2d(x, inv_filters_exp, stride=2)
        res[:, i:i+1] = temp

    return res


class WaveletEncoderV2(nn.Module):
    def __init__(self, base_channels=64, num_downsample=1, dropout_prob=0.3):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.num_downsample = num_downsample
        self.dropout_prob = dropout_prob

        # LL path with 1 BatchNorm after last conv
        self.conv_ll = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # HF path with 1 BatchNorm after last conv
        self.conv_hf = nn.Sequential(
            nn.Conv2d(9, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
        )

        in_channels = base_channels + base_channels // 2 + 1  # LL + HF + init_mask
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
        Yl, Yh = self.dwt(rgb)

        if Yh[0].dim() == 5:
            hf = Yh[0].permute(0, 2, 1, 3, 4).reshape(Yh[0].shape[0], -1, Yh[0].shape[3], Yh[0].shape[4])
        else:
            hf = Yh[0]

        feat_ll = self.conv_ll(Yl)
        feat_hf = self.conv_hf(hf)
        mask_ds = F.interpolate(init_mask, size=feat_ll.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([feat_ll, feat_hf, mask_ds], dim=1)
        fused = self.fuse(x)

        # 保存中间结果（for decoder）
        self.cached_feats = {
            'll': Yl,
            'hf': hf,
            'feat_ll': feat_ll,
            'feat_hf': feat_hf,
        }

        return fused


class WaveletDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, base_channels, dropout_prob=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
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
        # x: deep latent
        # encoder_feats: dict with 'll', 'hf', 'feat_ll', 'feat_hf'
        ll = encoder_feats['ll']
        hf = encoder_feats['hf']

        # 拼回输入给 iwt: [B, 4C, H, W]
        hf_split = torch.chunk(hf, 3, dim=1)
        fused = torch.cat([ll, hf_split[0], hf_split[1], hf_split[2]], dim=1)
        x = iwt(fused)

        skip_feat = torch.cat([encoder_feats['feat_ll'], encoder_feats['feat_hf']], dim=1)
        skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        error_map = F.interpolate(error_map, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = self.proj(x)
        x = torch.cat([x, skip_feat, error_map], dim=1)
        return self.refine(x)



class RefinerWithDualBranch(nn.Module):
    def __init__(self, base_channels=64, num_downsample=1, dropout_prob=0.3):
        super().__init__()
        self.encoder = WaveletEncoderV2(
            base_channels=base_channels,
            num_downsample=num_downsample,
            dropout_prob=dropout_prob
        )

        self.skip_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
            # Dropout intentionally removed to reduce GPU cost
        )

        # Decoder expects iwt() output with 3 channels (RGB reconstructed image)
        in_channels = 3
        skip_channels = base_channels + base_channels // 2

        self.fp_decoder = WaveletDecoder(
            in_channels=in_channels,
            skip_channels=skip_channels,
            base_channels=base_channels,
            dropout_prob=dropout_prob
        )
        self.fn_decoder = WaveletDecoder(
            in_channels=in_channels,
            skip_channels=skip_channels,
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
            correction_fp = correction_fp * fp_mask
        if fn_mask is not None:
            correction_fn = correction_fn * fn_mask

        final_pred = (init_mask - correction_fp + correction_fn).clamp(0, 1)
        return final_pred, correction_fp, correction_fn

