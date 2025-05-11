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

# === DWT / IWT ===
def custom_dwt(vimg):
    B, C, H, W = vimg.shape
    assert H % 2 == 0 and W % 2 == 0
    res = torch.zeros(B, 4*C, H//2, W//2, device=vimg.device)
    for i in range(C):
        res[:, 4*i:4*i+4] = F.conv2d(vimg[:, i:i+1], filters.to(vimg.device), stride=2)
        res[:, 4*i+1:4*i+4] = (res[:, 4*i+1:4*i+4] + 1) / 2.0
    ll = res[:, 0::4]
    hf = torch.cat([res[:, 4*i+1:4*i+4] for i in range(C)], dim=1)
    return ll, hf, {"orig_shape": (H, W)}

def custom_iwt(ll, hf):
    B, C, H, W = ll.shape
    res = torch.zeros(B, 4*C, H, W, device=ll.device)
    for i in range(C):
        res[:, 4*i] = ll[:, i]
        res[:, 4*i+1:4*i+4] = hf[:, 3*i:3*i+3]
        res[:, 4*i+1:4*i+4] = 2 * res[:, 4*i+1:4*i+4] - 1
    recon = torch.zeros(B, C, H*2, W*2, device=ll.device)
    for i in range(C):
        recon[:, i:i+1] = F.conv_transpose2d(
            res[:, 4*i:4*i+4],
            inv_filters.to(ll.device),
            stride=2
        )
    return recon

# === Encoder ===
class WaveletEncoderV2(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.3):
        super().__init__()
        self.mask_down_once = nn.Sequential(
            nn.Conv2d(1, base_channels//2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.error_down_once = nn.Sequential(
            nn.Conv2d(1, base_channels//2, 3, stride=2, padding=1),
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
            nn.Conv2d(3*3, base_channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, base_channels//2, 3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels + base_channels//2 + base_channels//2,
                      base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)
        )

    def compute_required_downsample_steps(self, h_input, h_target):
        ratio = h_input // h_target
        return int(torch.log2(torch.tensor(ratio)).item())

    def forward(self, rgb, init_mask, error_map):
        # wavelet decomposition
        ll, hf, info = custom_dwt(rgb)
        feat_ll = self.conv_ll(ll)
        feat_hf = self.conv_hf(hf)
        # downsample mask & error to match
        H_in, H_feat = init_mask.shape[2], feat_ll.shape[2]
        steps = self.compute_required_downsample_steps(H_in, H_feat)
        mask_ds, error_ds = init_mask, error_map
        for _ in range(steps):
            mask_ds  = self.mask_down_once(mask_ds)
            error_ds = self.error_down_once(error_ds)
        x = torch.cat([feat_ll, feat_hf, mask_ds + error_ds], dim=1)
        fused = self.fuse(x)
        return fused, ll, hf, info

# === Decoder ===
class DecoderBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.recover_conv = nn.Sequential(
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, fused, ll, hf, info):
        refined_ll = self.recover_conv(fused)
        return custom_iwt(refined_ll, hf)

# === Refiner with Residual Fusion & LL‐Gating ===
class RefinerWithResidualGating(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.3):
        super().__init__()
        self.encoder      = WaveletEncoderV2(base_channels, dropout_prob)
        self.skip_conv    = nn.Conv2d(base_channels*2, base_channels*2, 1)
        self.fusion_conv  = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.base_decoder = DecoderBlock(base_channels*2)

        # matte heads
        self.to_mask          = nn.Conv2d(3, 1, 1)
        self.to_mask_unguided = nn.Conv2d(3, 1, 1)

        # gate head: take fused + (optional) error_map
        c = base_channels*2
        self.gate_head = nn.Sequential(
            nn.Conv2d(c + 1, base_channels, 3, padding=1),  # +1 if concat error_map_ds
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
        )

    def forward(self, rgb, init_mask, error_map):
        # 1) encode guided
        fused_feat, ll_guided, hf, info = self.encoder(rgb, init_mask, error_map)
        # 2) encode unguided low-freq only
        with torch.no_grad():
            _, ll_unguided, _, _ = self.encoder(rgb, torch.zeros_like(init_mask), error_map)

        # 3) fusion conv + skip
        skip  = self.skip_conv(fused_feat)
        fused = self.fusion_conv(torch.cat([fused_feat, skip], dim=1))

        # 4) build gate input: fused + error_map_ds
        error_map_ds = F.interpolate(
            error_map, size=fused.shape[-2:], mode='bilinear', align_corners=False
        )
        gate_logits = self.gate_head(torch.cat([fused, error_map_ds], dim=1))
        gate = torch.sigmoid(gate_logits)

        # 5) upsample gates
        gate_pix = F.interpolate(
            gate, size=init_mask.shape[-2:], mode='bilinear', align_corners=False
        )
        gate_ll = F.interpolate(
            gate, size=ll_guided.shape[-2:], mode='bilinear', align_corners=False
        )

        # 6) decode intermediate (guided low-freq + hf)
        decoded = self.base_decoder(fused, ll_guided, hf, info)
        mg = torch.sigmoid(self.to_mask(decoded))
        mu = torch.sigmoid(self.to_mask_unguided(decoded))

        # 7) pixel-domain residual fusion
        refined_pix = mg + gate_pix * (mu - mg)

        # 8) LL-domain residual fusion
        ll_refined = ll_guided + gate_ll * (ll_unguided - ll_guided)

        # 9) final decode from ll_refined + hf
        final_decoded = self.base_decoder(fused, ll_refined, hf, info)
        refined = torch.sigmoid(self.to_mask(final_decoded))

        return refined, mg, mu, gate
