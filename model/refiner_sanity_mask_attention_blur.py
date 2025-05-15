#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
refiner_crossband.py  ──────────────────────────────────────────────
Wavelet‑based foreground matte **Refiner** with **HF Cross‑Band
Self‑Attention (standard learnable Q/K/V)**.

Key points
~~~~~~~~~~
* **DWT multi‑scale encoder / decoder** + *Residual Gate* keep the
  original behaviour unchanged.
* For each DWT level we insert a lightweight 3‑token Self‑Attention
  block that explicitly exchanges information between the LH / HL /
  HH high‑frequency bands (→ sequence length = 3).
* The attention runs **before** the existing `conv_hf`, so channel
  dimensions stay the same (3×in_channels).

Usage: simply import `RefinerMixed` from this file in place of the
previous implementation.
"""
from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pywt

# -----------------------------------------------------------------------------
# Wavelet utilities (unchanged) ------------------------------------------------
# -----------------------------------------------------------------------------
SUPPORTED_WAVELETS = ["db1", "db2", "db4"]

wavelet_filters, wavelet_inv_filters = {}, {}
for name in SUPPORTED_WAVELETS:
    w = pywt.Wavelet(name)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
    rec_hi = torch.tensor(w.rec_hi,      dtype=torch.float32)
    rec_lo = torch.tensor(w.rec_lo,      dtype=torch.float32)

    filt = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1) / 2.0,
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)[:, None]

    inv = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1) * 2.0,
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)[:, None]

    wavelet_filters[name]     = Variable(filt, requires_grad=False)
    wavelet_inv_filters[name] = Variable(inv,  requires_grad=False)


def dwt_level(x: torch.Tensor, wname: str):
    filt = wavelet_filters[wname].to(x.device)
    B, C, H, W = x.shape
    pad = (filt.size(-1) - 1) // 2
    out = []
    for i in range(C):
        xi = F.pad(x[:, i:i+1], (pad, pad, pad, pad), mode="reflect")
        coeff = F.conv2d(xi, filt, stride=2)
        out.append(coeff)
    out = torch.cat(out, 1)
    ll = out[:, 0::4]
    hf = torch.cat([out[:, 4*i+1:4*i+4] for i in range(C)], 1)
    # hf = (hf + 1) / 2
    return ll, hf


def iwt_level(ll: torch.Tensor, hf: torch.Tensor, wname: str):
    # hf = hf * 2 - 1
    inv = wavelet_inv_filters[wname].to(ll.device)
    B, C, H, W = ll.shape
    pad = (inv.size(-1) - 1) // 2
    recs = []
    for i in range(C):
        coeff = torch.cat([ll[:, i:i+1], hf[:, 3*i:3*i+3]], 1)
        rec = F.conv_transpose2d(coeff, inv, stride=2)
        if pad:
            rec = F.pad(rec, (pad, pad, pad, pad), mode="reflect")
            rec = rec[..., 2*pad:-2*pad, 2*pad:-2*pad]
        recs.append(rec)
    return torch.cat(recs, 1)


def multi_dwt_mixed(x: torch.Tensor, wlist):
    coeffs, cur = [], x
    for w in wlist:
        ll, hf = dwt_level(cur, w)
        coeffs.append((ll, hf, w))
        cur = ll
    return coeffs


def multi_iwt_mixed(coeffs):
    cur = coeffs[-1][0]
    for ll, hf, w in reversed(coeffs):
        cur = iwt_level(cur, hf, w)
    return cur

class HybridBandAttn(nn.Module):
    def __init__(self, in_channels: int, heads: int = 4):
        super().__init__()
        # in_channels = ll_channels + hf_channels
        assert in_channels % 4 == 0, "in_channels must be 4×k for full-band"
        self.band_dim = in_channels // 4
        self.heads = min(heads, self.band_dim)
        while self.band_dim % self.heads != 0:
            self.heads -= 1
        self.head_dim = self.band_dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.res_scale = nn.Parameter(torch.tensor(0.5))

        self.qkv = nn.Linear(self.band_dim, 3*self.band_dim, bias=False)
        self.proj = nn.Linear(self.band_dim, self.band_dim)

    def forward(self, ll: torch.Tensor, hf: torch.Tensor):
        B, C, H, W = ll.shape
        hf_split = hf.view(B, 3, C, H, W)
        x = torch.cat([ll.unsqueeze(1), hf_split], dim=1)  # (B,4,C,H,W)
        x_flat = x.permute(0,3,4,1,2).reshape(-1, 4, C)

        qkv = self.qkv(x_flat).chunk(3, dim=-1)
        q, k, v = [t.reshape(-1,4,self.heads,self.head_dim).transpose(1,2) for t in qkv]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        y = (attn @ v).transpose(1,2).reshape(-1,4,C)
        y = self.proj(y)

        y = y.view(B, H, W, 4, C).permute(0,3,4,1,2).reshape(B,4*C,H,W)
        return x.view(B,4*C,H,W) + self.res_scale * y

class HFOnlyAttn(nn.Module):
    def __init__(self, in_channels: int, heads: int = 4):
        super().__init__()
        assert in_channels % 3 == 0, "in_channels must be 3×k for HF-only"
        self.band_dim = in_channels // 3
        self.heads = min(heads, self.band_dim)
        while self.band_dim % self.heads != 0:
            self.heads -= 1
        self.head_dim = self.band_dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.res_scale = nn.Parameter(torch.tensor(0.5))

        self.qkv = nn.Linear(self.band_dim, 3*self.band_dim, bias=False)
        self.proj = nn.Linear(self.band_dim, self.band_dim)

    def forward(self, hf: torch.Tensor):
        B, Ctot, H, W = hf.shape
        bd = self.band_dim
        x = hf.view(B,3,bd,H,W).permute(0,3,4,1,2).reshape(-1,3,bd)

        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(-1,3,self.heads,self.head_dim).transpose(1,2) for t in qkv]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        y = (attn @ v).transpose(1,2).reshape(-1,3,bd)
        y = self.proj(y)
        y = y.view(B,H,W,3,bd).permute(0,3,4,1,2).reshape(B,Ctot,H,W)
        return hf + self.res_scale * y

# -----------------------------------------------------------------------------
# Encoder ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
class WaveletEncoderHybrid(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.3,
                 wavelet_list: List[str]|None=None, in_channels=3,
                 err_channels=1, heads=4):
        super().__init__()
        assert wavelet_list, "wavelet_list must not be empty"
        self.wavelet_list = wavelet_list
        self.levels = len(wavelet_list)

        self.mask_down = nn.MaxPool2d(2,2,ceil_mode=True)
        self.err_down  = nn.AvgPool2d(2)

        self.conv_ll = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, padding_mode="reflect"), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(base_channels), nn.ReLU()
        )
        self.conv_hf = nn.Sequential(
            nn.Conv2d(3*in_channels, base_channels, 3, padding=1, padding_mode="reflect"), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(base_channels), nn.ReLU()
        )

        # 两路注意力分支
        self.hf_attn   = nn.ModuleList([HFOnlyAttn(3*in_channels, heads) for _ in range(self.levels)])
        self.fb_attn   = nn.ModuleList([HybridBandAttn(in_channels + 3*in_channels, heads) for _ in range(self.levels)])
        # gate 融合维度 = base_channels * 2
        self.attn_gate = nn.ModuleList([nn.Conv2d(base_channels*2, 1, 1) for _ in range(self.levels)])

        in_ch_fuse = self.levels*(base_channels*2) + 1 + err_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch_fuse, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, rgb: torch.Tensor, init_mask: torch.Tensor, err_map: torch.Tensor):
        coeffs = multi_dwt_mixed(rgb, self.wavelet_list)
        ll_feats, hf_feats = [], []
        for i, (ll, hf, w) in enumerate(coeffs):
            # LL 分支
            ll_f = self.conv_ll(ll)

            # HF-only 分支
            hf_h = self.hf_attn[i](hf)
            hf_f = self.conv_hf(hf_h)  # -> [B, base_channels, H, W]

            # Full-band 分支
            fb_out = self.fb_attn[i](ll, hf)  # -> [B, ll_C + hf_C, H, W]
            _, hf_fb_raw = fb_out.split((ll.shape[1], hf.shape[1]), dim=1)
            hf_fb = self.conv_hf(hf_fb_raw)   # -> [B, base_channels, H, W]

            # 融合 HF 特征
            gate = torch.sigmoid(self.attn_gate[i](torch.cat([hf_f, hf_fb], dim=1)))  # -> [B,1,H,W]
            hf_comb = hf_f * gate + hf_fb * (1 - gate)  # -> [B, base_channels, H, W]

            ll_feats.append(ll_f)
            hf_feats.append(hf_comb)

        # 对齐所有尺度特征并融合
        tgt_h, tgt_w = ll_feats[-1].shape[-2:]
        aligned = []
        for f in ll_feats + hf_feats:
            while f.shape[-2] > tgt_h:
                f = F.avg_pool2d(f, 2)
            aligned.append(f)
        m, e = init_mask, err_map
        for _ in range(self.levels):
            m = self.mask_down(m)
            e = self.err_down(e)
        aligned.extend([m, e])
        fused = self.fuse(torch.cat(aligned, 1))
        return fused, coeffs


# -----------------------------------------------------------------------------
# Decoder ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch:int, hf_ch:int, init_scale:float=0.1):
        super().__init__()
        self.hf_scale = nn.Parameter(torch.tensor(init_scale))
        self.recov_ll = nn.Sequential(nn.Conv2d(in_ch,3,3,padding=1),nn.Tanh())
        self.recov_hf = nn.Sequential(nn.Conv2d(in_ch,hf_ch,3,padding=1),nn.Tanh())
    def forward(self,fused,coeffs):
        ll,hf,w = coeffs[-1]
        ll_r = self.recov_ll(fused)
        hf_r = self.recov_hf(fused)*torch.abs(self.hf_scale)
        hf_new = (hf + hf_r).clamp(-1,1)
        coeffs[-1] = (ll_r,hf_new,w)
        return multi_iwt_mixed(coeffs)

class RefinerMixedHybrid(nn.Module):
    def __init__(self, base_channels:int=64, dropout_prob:float=0.3,
                 wavelet_list:List[str]|None=None, in_channels:int=3,
                 err_channels:int=2, heads:int=4):
        super().__init__()
        self.encoder = WaveletEncoderHybrid(base_channels,dropout_prob,wavelet_list,
                                            in_channels,err_channels,heads)
        c2 = base_channels*2; hf_ch = 3*in_channels
        self.decoder = DecoderBlock(c2,hf_ch)
        self.to_mask   = nn.Conv2d(3,1,1)
        self.to_mask_u = nn.Conv2d(3,1,1)
        self.gate_head = nn.Sequential(
            nn.Conv2d(c2+err_channels,base_channels,3,padding=1),nn.ReLU(),
            nn.Conv2d(base_channels,base_channels,3,padding=1),nn.ReLU(),
            nn.Conv2d(base_channels,1,1)
        )
        self.up = nn.ConvTranspose2d(1,1,2,2)

    def forward(self,rgb,init_mask,err_map):
        fused,coeffs_g = self.encoder(rgb,init_mask,err_map)
        _,coeffs_u = self.encoder(rgb,torch.zeros_like(init_mask),err_map)
        e_ds = err_map
        for _ in range(self.encoder.levels): e_ds = self.encoder.err_down(e_ds)
        gate_d = torch.sigmoid(self.gate_head(torch.cat([fused,e_ds],1)))
        gate_p = gate_d
        for _ in range(self.encoder.levels): gate_p = self.up(gate_p)
        dec_g = self.decoder(fused,coeffs_g); mg = torch.sigmoid(self.to_mask(dec_g))
        dec_u = self.decoder(fused,coeffs_u); mu = torch.sigmoid(self.to_mask_u(dec_u))
        refined = mg + gate_p*(mu-mg)
        return refined,mg,mu,gate_d

# -----------------------------------------------------------------------------
#  Quick self‑test  ────────────────────────────────────────────────────────────
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2,3,256,256); m0 = torch.rand(2,1,256,256); err = torch.rand(2,2,256,256)
    model = RefinerMixedHybrid(base_channels=32,dropout_prob=0.0,
                               wavelet_list=['db4','db2','db1'],err_channels=2,heads=4)
    with torch.no_grad(): rf,mg,mu,gate = model(x,m0,err)
    print("PASS", rf.shape, mg.shape, mu.shape, gate.shape)
