# refiner_wavelet_fix.py
# -------------------------------------------------------------
# Wavelet‑based image refiner ("mask‑as‑guide" variant)
# 修复 BUG：
#   1) 正 / 逆小波变换边界延拓方式不一致 → “丢边”
#   2) Decoder HF 残差通道数错误 (3→9)
#   3) Fuse, Gate 通道数与 ErrMap 维度对齐
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from torch.autograd import Variable

# ------------------------------------------------------------------
#  Wavelet filter banks（支持 db1 / db2 / db4，可自行扩展）
# ------------------------------------------------------------------
SUPPORTED_WAVELETS = ["db1", "db2", "db4"]

wavelet_filters, wavelet_inv_filters = {}, {}
for name in SUPPORTED_WAVELETS:
    w = pywt.Wavelet(name)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
    rec_hi = torch.tensor(w.rec_hi,      dtype=torch.float32)
    rec_lo = torch.tensor(w.rec_lo,      dtype=torch.float32)

    # forward DWT filters: LL, LH, HL, HH
    filt = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1) / 2.0,
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)[:, None]

    # inverse IWT filters
    inv = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)[:, None]

    wavelet_filters[name]     = Variable(filt, requires_grad=False)
    wavelet_inv_filters[name] = Variable(inv,  requires_grad=False)

# ------------------------------------------------------------------
#  Single‑level DWT / IWT
# ------------------------------------------------------------------

def dwt_level(x: torch.Tensor, wavelet_name: str):
    """Single‑level forward DWT using reflect padding."""
    filt = wavelet_filters[wavelet_name].to(x.device)
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "Input size must be divisible by 2"
    k = filt.size(-1)
    pad = (k - 1) // 2

    out = x.new_zeros(B, 4 * C, H // 2, W // 2)
    for i in range(C):
        xi_p = F.pad(x[:, i:i+1], (pad, pad, pad, pad), mode='reflect')
        coeffs = F.conv2d(xi_p, filt, stride=2)
        out[:, 4 * i] = coeffs[:, 0]
        out[:, 4 * i + 1: 4 * i + 4] = (coeffs[:, 1:] + 1) / 2

    ll = out[:, 0::4]
    hf = torch.cat([out[:, 4 * i + 1: 4 * i + 4] for i in range(C)], dim=1)
    return ll, hf

def iwt_level(ll: torch.Tensor, hf: torch.Tensor, wavelet_name: str):
    """Single‑level inverse transform with aligned border handling."""
    inv = wavelet_inv_filters[wavelet_name].to(ll.device)
    B, C, H, W = ll.shape
    k = inv.size(-1)
    pad = (k - 1) // 2

    res = ll.new_zeros(B, 4 * C, H, W)
    for i in range(C):
        res[:, 4 * i] = ll[:, i]
        res[:, 4 * i + 1: 4 * i + 4] = hf[:, 3 * i: 3 * i + 3] * 2 - 1

    outs = []
    for i in range(C):
        ct = F.conv_transpose2d(res[:, 4 * i: 4 * i + 4], inv, stride=2)
        if pad:
            ct = F.pad(ct, (pad, pad, pad, pad), mode='reflect')
            ct = ct[..., 2*pad:-2*pad, 2*pad:-2*pad]
        outs.append(ct)
    return torch.cat(outs, dim=1)

# ------------------------------------------------------------------
#  Multi‑level helpers
# ------------------------------------------------------------------

def multi_dwt_mixed(x: torch.Tensor, wavelet_list):
    coeffs = []
    cur = x
    for w in wavelet_list:
        ll, hf = dwt_level(cur, w)
        coeffs.append((ll, hf, w))
        cur = ll
    return coeffs

def multi_iwt_mixed(coeffs):
    cur = coeffs[-1][0]
    for ll, hf, w in reversed(coeffs):
        cur = iwt_level(cur, hf, w)
    return cur

# ------------------------------------------------------------------
#  Encoder / Decoder / Refiner
# ------------------------------------------------------------------

class WaveletEncoderMixed(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.3,
                 wavelet_list=None, in_channels=3,
                 err_channels: int = 1):
        super().__init__()
        assert wavelet_list, "empty wavelet_list"
        self.wavelet_list = wavelet_list
        self.levels = len(wavelet_list)
        self.err_channels = err_channels
        self.mask_down = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.err_down = nn.AvgPool2d(2)

        self.conv_ll = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(base_channels), nn.ReLU()
        )
        self.conv_hf = nn.Sequential(
            nn.Conv2d(3 * in_channels, base_channels // 2, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1, padding_mode='reflect'), nn.BatchNorm2d(base_channels // 2), nn.ReLU()
        )

        in_ch_fuse = self.levels * (base_channels + base_channels // 2) + 1 + err_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch_fuse, base_channels * 2, 3, padding=1), nn.BatchNorm2d(base_channels * 2), nn.ReLU(),
            nn.Dropout2d(dropout_prob)
        )

    def forward(self, rgb, init_mask, err_map):
        coeffs = multi_dwt_mixed(rgb, self.wavelet_list)
        ll_feats = [self.conv_ll(ll) for ll, _, _ in coeffs]
        hf_feats = [self.conv_hf(hf) for _, hf, _ in coeffs]

        target_h, target_w = ll_feats[-1].shape[-2:]
        aligned = []
        for f in ll_feats + hf_feats:
            while f.shape[-2] > target_h:
                f = F.avg_pool2d(f, 2)
            aligned.append(f)

        m, e = init_mask, err_map
        for _ in range(self.levels):
            m = self.mask_down(m)
            e = self.err_down(e)
        aligned += [m, e]

        fused = self.fuse(torch.cat(aligned, 1))
        return fused, coeffs

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, hf_ch, init_scale=0.1):
        super().__init__()
        self.hf_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.recov_ll = nn.Sequential(nn.Conv2d(in_ch, 3, 3, padding=1, padding_mode='reflect'), nn.Tanh())
        self.recov_hf = nn.Sequential(nn.Conv2d(in_ch, hf_ch, 3, padding=1, padding_mode='reflect'), nn.Tanh())

    def forward(self, fused, coeffs):
        ll, hf, w = coeffs[-1]
        ll_r = self.recov_ll(fused)
        hf_r = self.recov_hf(fused) * torch.abs(self.hf_scale)
        hf_ref = (hf * 2 - 1 + hf_r).clamp(-1, 1)
        coeffs[-1] = (ll_r, (hf_ref + 1) / 2, w)
        return multi_iwt_mixed(coeffs)

class RefinerMixed(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.3,
                 wavelet_list=None, in_channels=3,
                 err_channels: int = 2):
        super().__init__()
        assert wavelet_list, "empty wavelet_list"

        self.encoder = WaveletEncoderMixed(base_channels, dropout_prob,
                                           wavelet_list, in_channels,
                                           err_channels=err_channels)
        c2 = base_channels * 2
        hf_ch = 3 * in_channels  # 9 for RGB
        self.decoder = DecoderBlock(c2, hf_ch)

        self.to_mask   = nn.Conv2d(3, 1, 1)
        self.to_mask_u = nn.Conv2d(3, 1, 1)

        self.gate_head = nn.Sequential(
            nn.Conv2d(c2 + err_channels, base_channels, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, padding_mode='reflect'), nn.ReLU(),
            nn.Conv2d(base_channels, 1, 1)
        )
        self.up = nn.ConvTranspose2d(1, 1, 2, 2)

    def forward(self, rgb, init_mask, err_map):
        fused, coeffs_g = self.encoder(rgb, init_mask, err_map)
        _, coeffs_u = self.encoder(rgb, torch.zeros_like(init_mask), err_map)

        e_ds = err_map
        for _ in range(self.encoder.levels):
            e_ds = self.encoder.err_down(e_ds)

        gate_d = torch.sigmoid(self.gate_head(torch.cat([fused, e_ds], 1)))
        gate_p = gate_d
        for _ in range(self.encoder.levels):
            gate_p = self.up(gate_p)

        dec_g = self.decoder(fused, coeffs_g)
        mg = torch.sigmoid(self.to_mask(dec_g))
        dec_u = self.decoder(fused, coeffs_u)
        mu = torch.sigmoid(self.to_mask_u(dec_u))

        refined = mg + gate_p * (mu - mg)
        return refined, mg, mu, gate_d

# ------------------------------------------------------------------
#  Quick self‑test --------------------------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    img = torch.randn(2, 3, 256, 256)
    mask = torch.rand(2, 1, 256, 256)
    err = torch.rand(2, 2, 256, 256)

    model = RefinerMixed(base_channels=32, wavelet_list=["db2"],
                         dropout_prob=0.0, err_channels=2)
    with torch.no_grad():
        out, mg, mu, gate = model(img, mask, err)
    print("PASS", out.shape, mg.shape, gate.shape)