import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


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
        return fused


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
    def __init__(self, base_channels=64, num_downsample=1, dropout_prob=0.3):
        super().__init__()
        self.encoder = WaveletEncoderV2(base_channels=base_channels, num_downsample=num_downsample, dropout_prob=dropout_prob)

        self.skip_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
            # Dropout removed here to reduce GPU cost
        )

        in_channels = base_channels * 2
        num_upsample = num_downsample + 1
        self.fp_decoder = DecoderBlock(in_channels, base_channels, num_upsample=num_upsample, dropout_prob=dropout_prob)
        self.fn_decoder = DecoderBlock(in_channels, base_channels, num_upsample=num_upsample, dropout_prob=dropout_prob)

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

        assert correction_fp.shape == init_mask.shape, \
            f"Correction_fp shape {correction_fp.shape} does not match init_mask {init_mask.shape}"
        assert correction_fn.shape == init_mask.shape, \
            f"Correction_fn shape {correction_fn.shape} does not match init_mask {init_mask.shape}"

        final_pred = (init_mask - correction_fp + correction_fn).clamp(0, 1)
        return final_pred, correction_fp, correction_fn
