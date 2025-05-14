import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from torch.autograd import Variable

# === Config: choose wavelet ===
# Supported: 'db1', 'db2', 'db4'
wavelet_name = 'db4'

# === Wavelet Filters ===
w = pywt.Wavelet(wavelet_name)
dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)

# compute pad size for reflect padding
filter_length = dec_lo.numel()
pad = (filter_length - 1) // 2

# build filter banks
filters = torch.stack([
    dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1) / 2.0,
    dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
    dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
    dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
], dim=0)[:, None]
inv_filters = torch.stack([
    rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,
    rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
], dim=0)[:, None]
filters = Variable(filters, requires_grad=False)
inv_filters = Variable(inv_filters, requires_grad=False)

# === Single-level DWT / IWT with manual reflect padding ===

def custom_dwt(vimg):
    B, C, H, W = vimg.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
    out = torch.zeros(B, 4*C, H//2, W//2, device=vimg.device)
    for i in range(C):
        # manual reflect padding before convolution
        x = vimg[:, i:i+1]
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        tmp = F.conv2d(x_pad, filters.to(vimg.device), stride=2)
        # combine LL and HF (HF encoded in [0,1] mapped from [-1,1])
        out[:, 4*i:4*i+4] = torch.cat([tmp[:, :1], (tmp[:, 1:] + 1) / 2], dim=1)
    ll = out[:, 0::4]
    hf = torch.cat([out[:, 4*i+1:4*i+4] for i in range(C)], dim=1)
    return ll, hf


def custom_iwt(ll, hf):
    B, C, H, W = ll.shape
    # interleave subbands
    res = torch.zeros(B, 4*C, H, W, device=ll.device)
    for i in range(C):
        res[:, 4*i] = ll[:, i]
        res[:, 4*i+1:4*i+4] = hf[:, 3*i:3*i+3] * 2 - 1
    # transposed convolution for upsampling
    recon = torch.zeros(B, C, H*2, W*2, device=ll.device)
    for i in range(C):
        recon[:, i:i+1] = F.conv_transpose2d(
            res[:, 4*i:4*i+4], inv_filters.to(ll.device),
            stride=2, padding=pad, output_padding=0
        )
    return recon

# === Multi-scale DWT/IWT helpers ===
def multi_dwt(x, J):
    coeffs, curr = [], x
    for _ in range(J):
        ll, hf = custom_dwt(curr)
        coeffs.append((ll, hf))
        curr = ll
    return coeffs


def multi_iwt(coeffs):
    curr = coeffs[-1][0]
    for ll, hf in reversed(coeffs):
        curr = custom_iwt(curr, hf)
    return curr

# === Encoder: Cascade DWT only ===
class WaveletEncoderCascade(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.3, levels=2):
        super().__init__()
        self.levels = levels
        self.mask_down = nn.AvgPool2d(2)
        self.err_down = nn.AvgPool2d(2)

        # LL branch
        self.conv_ll = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU()
        )
        # HF branch (for all channels combined: 3 channels * 3 subbands)
        self.conv_hf = nn.Sequential(
            nn.Conv2d(9, base_channels//2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels//2, base_channels//2, 3, padding=1),
            nn.BatchNorm2d(base_channels//2), nn.ReLU()
        )

        in_ch = levels * base_channels + levels * (base_channels//2) + 1 + 2
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2), nn.ReLU(),
            nn.Dropout2d(dropout_prob)
        )

    def forward(self, rgb, init_mask, err_map):
        coeffs = multi_dwt(rgb, self.levels)

        feat_ll, feat_hf = [], []
        for ll, hf in coeffs:
            feat_ll.append(self.conv_ll(ll))
            feat_hf.append(self.conv_hf(hf))

        # align to smallest resolution
        th, tw = feat_ll[-1].shape[-2:]
        aligned = []
        for f in feat_ll + feat_hf:
            curr = f
            while curr.shape[-2] > th:
                curr = F.avg_pool2d(curr, 2)
            aligned.append(curr)

        # downsample init_mask & err_map
        m, e = init_mask, err_map
        for _ in range(self.levels):
            m = self.mask_down(m)
            e = self.err_down(e)
        aligned += [m, e]

        x = torch.cat(aligned, dim=1)
        fused = self.fuse(x)
        return fused, coeffs

# === Decoder ===
class DecoderBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.recov = nn.Sequential(
            nn.Conv2d(in_ch, 3, 3, padding=1), nn.Tanh()
        )

    def forward(self, fused, coeffs):
        ll, hf = coeffs[-1]
        ll_ref = self.recov(fused)
        coeffs[-1] = (ll_ref, hf)
        return multi_iwt(coeffs)

# === Refiner: Cascade only ===
class RefinerCascade(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.3, levels=2):
        super().__init__()
        self.encoder = WaveletEncoderCascade(base_channels, dropout_prob, levels)
        c = base_channels*2
        self.decoder = DecoderBlock(c)
        self.to_mask = nn.Conv2d(3, 1, 1)
        self.to_mask_unguided = nn.Conv2d(3, 1, 1)

        self.gate_head = nn.Sequential(
            nn.Conv2d(c+2, base_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(base_channels, 1, 1)
        )
        self.up = nn.ConvTranspose2d(1, 1, 2, 2)

    def forward(self, rgb, init_mask, err_map):
        fused, coeffs = self.encoder(rgb, init_mask, err_map)

        e_ds = err_map
        for _ in range(self.encoder.levels):
            e_ds = self.encoder.err_down(e_ds)
        gate = torch.sigmoid(self.gate_head(torch.cat([fused, e_ds], dim=1)))

        g_pix = gate
        for _ in range(self.encoder.levels):
            g_pix = self.up(g_pix)

        dec_g = self.decoder(fused, coeffs)
        mg = torch.sigmoid(self.to_mask(dec_g))

        _, coeffs_u = self.encoder(rgb, torch.zeros_like(init_mask), err_map)
        dec_u = self.decoder(fused, coeffs_u)
        mu = torch.sigmoid(self.to_mask_unguided(dec_u))

        refined = mg + g_pix * (mu - mg)
        return refined, mg, mu, gate

