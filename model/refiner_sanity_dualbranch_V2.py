import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from torch.autograd import Variable

# === Wavelet Filters ===
w = pywt.Wavelet('db1')
dec_hi = torch.Tensor(w.dec_hi[::-1])
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

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

filters = Variable(filters[:, None], requires_grad=False)
inv_filters = Variable(inv_filters[:, None], requires_grad=False)

# === DWT ===
def custom_dwt(vimg):
    B, C, H, W = vimg.shape
    assert H % 2 == 0 and W % 2 == 0

    res = torch.zeros(B, 4 * C, H // 2, W // 2).to(vimg.device)
    for i in range(C):
        res[:, 4 * i:4 * i + 4] = F.conv2d(vimg[:, i:i + 1], filters.to(vimg.device), stride=2)
        res[:, 4 * i + 1:4 * i + 4] = (res[:, 4 * i + 1:4 * i + 4] + 1) / 2.0

    ll = res[:, 0::4]
    hf = torch.cat([res[:, 4 * i + 1:4 * i + 4] for i in range(C)], dim=1)
    return ll, hf, {"orig_shape": (H, W)}

# === IWT ===
def custom_iwt(ll, hf):
    B, C, H, W = ll.shape
    res = torch.zeros(B, 4 * C, H, W).to(ll.device)
    for i in range(C):
        res[:, 4 * i] = ll[:, i]
        res[:, 4 * i + 1:4 * i + 4] = hf[:, 3 * i:3 * i + 3]
        res[:, 4 * i + 1:4 * i + 4] = 2 * res[:, 4 * i + 1:4 * i + 4] - 1

    recon = torch.zeros(B, C, H * 2, W * 2).to(ll.device)
    for i in range(C):
        recon[:, i:i + 1] = F.conv_transpose2d(
            res[:, 4 * i:4 * i + 4],
            inv_filters.to(ll.device),
            stride=2
        )
    return recon

# === Encoder ===
class WaveletEncoderV2(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.3):
        super().__init__()

        self.mask_down_once = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_ll = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_hf = nn.Sequential(
            nn.Conv2d(9, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
        )

        in_channels = base_channels + base_channels // 2
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )

    def compute_required_downsample_steps(self, h_input, h_target):
        assert h_input % h_target == 0
        ratio = h_input // h_target
        steps = int(torch.log2(torch.tensor(ratio)).item())
        assert 2 ** steps == ratio
        return steps

    def forward(self, rgb, init_mask):
        ll, hf, dwt_info = custom_dwt(rgb)
        feat_ll = self.conv_ll(ll)
        feat_hf = self.conv_hf(hf)

        H_in = init_mask.shape[2]
        H_feat = feat_ll.shape[2]
        steps = self.compute_required_downsample_steps(H_in, H_feat)

        mask_ds = init_mask
        for _ in range(steps):
            mask_ds = self.mask_down_once(mask_ds)

        feat_ll_mod = feat_ll * (1 + 0.5 * mask_ds)
        x = torch.cat([feat_ll_mod, feat_hf], dim=1)
        fused = self.fuse(x)
        return fused, ll, hf, dwt_info

# === Decoder ===
class DecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.recover_conv = nn.Sequential(
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, fused, ll, hf, dwt_info):
        refined_ll = self.recover_conv(fused)
        assert refined_ll.shape == ll.shape, f"refined_ll: {refined_ll.shape}, ll: {ll.shape}"
        return custom_iwt(refined_ll, hf)

# === Refiner ===
class RefinerWithDualBranch(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.3):
        super().__init__()
        self.encoder = WaveletEncoderV2(base_channels, dropout_prob)
        self.skip_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.fp_decoder = DecoderBlock(base_channels * 2)
        self.fn_decoder = DecoderBlock(base_channels * 2)
        self.to_mask = nn.Conv2d(3, 1, 1)

    def forward(self, rgb, init_mask, error_map=None, fp_mask=None, fn_mask=None):
        fused_feat, ll, hf, dwt_info = self.encoder(rgb, init_mask)
        skip_feat = self.skip_conv(fused_feat)
        fused = self.fusion_conv(torch.cat([fused_feat, skip_feat], dim=1))

        correction_fp = self.fp_decoder(fused, ll, hf, dwt_info)
        correction_fn = self.fn_decoder(fused, ll, hf, dwt_info)
        correction_fp = self.to_mask(correction_fp)
        correction_fn = self.to_mask(correction_fn)

        if fp_mask is not None:
            correction_fp *= fp_mask
        if fn_mask is not None:
            correction_fn *= fn_mask

        final_pred = (init_mask - correction_fp + correction_fn).clamp(0, 1)
        return final_pred, correction_fp, correction_fn
