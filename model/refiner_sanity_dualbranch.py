import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class WaveletEncoderV2(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv_ll = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_hf = nn.Sequential(
            nn.Conv2d(9, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels + base_channels // 2 + 1, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, init_mask):
        Yl, Yh = self.dwt(rgb)
        ll = Yl

        if Yh[0].dim() == 5:
            hf = Yh[0].permute(0, 2, 1, 3, 4).reshape(Yh[0].shape[0], -1, Yh[0].shape[3], Yh[0].shape[4])
        else:
            hf = Yh[0]

        feat_ll = self.conv_ll(ll)
        feat_hf = self.conv_hf(hf)
        mask_ds = F.interpolate(init_mask, size=feat_ll.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([feat_ll, feat_hf, mask_ds], dim=1)
        fused = self.fuse(x)
        return fused


class RefinerWithDualBranch(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()

        self.encoder = WaveletEncoderV2(base_channels=base_channels)

        self.skip_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fp_decoder = nn.Sequential(
            nn.Conv2d(base_channels * 2 + 1, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        self.fn_decoder = nn.Sequential(
            nn.Conv2d(base_channels * 2 + 1, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb, init_mask, error_map, fp_mask=None, fn_mask=None):
        # Encode features
        enc_feat = self.encoder(rgb, init_mask)
        skip_feat = self.skip_conv(enc_feat)
        fused_feat = self.fusion_conv(torch.cat([enc_feat, skip_feat], dim=1))

        # Prepare decoder input
        error_map_ds = F.interpolate(error_map, size=fused_feat.shape[2:], mode='bilinear', align_corners=False)
        decoder_input = torch.cat([fused_feat, error_map_ds], dim=1)

        # Decode corrections
        correction_fp = self.fp_decoder(decoder_input)
        correction_fn = self.fn_decoder(decoder_input)

        # Upsample to match init_mask resolution
        correction_fp = self.upsample(correction_fp)
        correction_fn = self.upsample(correction_fn)

        # Mask corrections if masks are provided
        if fp_mask is not None:
            fp_mask = F.interpolate(fp_mask, size=correction_fp.shape[2:], mode='nearest')
            correction_fp = correction_fp * fp_mask

        if fn_mask is not None:
            fn_mask = F.interpolate(fn_mask, size=correction_fn.shape[2:], mode='nearest')
            correction_fn = correction_fn * fn_mask

        # Final prediction
        final_pred = (init_mask - correction_fp + correction_fn).clamp(0, 1)
        return final_pred, correction_fp, correction_fn
