import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import kornia

# ---------------------------
# 可微 DWT 模块
# ---------------------------
class DWT(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        lp = torch.tensor([1.0, 1.0]) / np.sqrt(2.0)
        hp = torch.tensor([1.0, -1.0]) / np.sqrt(2.0)
        ll = lp.unsqueeze(1) @ lp.unsqueeze(0)
        lh = lp.unsqueeze(1) @ hp.unsqueeze(0)
        hl = hp.unsqueeze(1) @ lp.unsqueeze(0)
        hh = hp.unsqueeze(1) @ hp.unsqueeze(0)
        for name, kern in [('ll', ll), ('lh', lh), ('hl', hl), ('hh', hh)]:
            self.register_buffer(name, kern.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B,C,H,W = x.shape
        f = {k: v.repeat(C,1,1,1) for k,v in [('ll',self.ll),('lh',self.lh),('hl',self.hl),('hh',self.hh)]}
        LL = F.conv2d(x, f['ll'], stride=2, groups=C)
        LH = F.conv2d(x, f['lh'], stride=2, groups=C)
        HL = F.conv2d(x, f['hl'], stride=2, groups=C)
        HH = F.conv2d(x, f['hh'], stride=2, groups=C)
        HF = torch.cat([LH,HL,HH], dim=1)
        return LL, HF

# ---------------------------
# 基础模块：Encoder / ASPP / Fusion / Decoder / ConvBlock
# ---------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x): return self.net(x)

class LightASPP(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.ReLU(inplace=True)
        )
        self.gp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(2*out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        size = x.shape[-2:]
        y1 = self.conv1(x)
        y2 = self.gp(x)
        y2 = F.interpolate(y2, size=size, mode='bilinear', align_corners=False)
        return self.project(torch.cat([y1,y2], dim=1))

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1,6,12,18), num_groups=16):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(min(num_groups,out_ch), out_ch),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        self.gp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(dilations)+1)*out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    def forward(self, x):
        size = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        y = self.gp(x)
        y = F.interpolate(y, size=size, mode='bilinear', align_corners=False)
        feats.append(y)
        return self.project(torch.cat(feats, dim=1))

class FusionModule(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x): return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.net = nn.Sequential(
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x): return self.net(self.up(x))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups,out_ch), out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x): return self.net(x)

# --------------------------------------------------------
# 窗口化跨频 + 定向注意力 (含 Padding & Crop 支持任意 window_size)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int)
    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int)
        H (int): original height
        W (int): original width
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, -1, H, W)
    return x


class WindowCrossBandDir(nn.Module):
    """
    Cross-band attention with overlapping windows and chunked processing.
    Splits high-frequency channels and applies SE, then overlapping-window QKV attention.
    """
    def __init__(self, ch, window_size=8, reduction=16, num_heads=4, chunk_size=64):
        super().__init__()
        self.window_size = window_size
        self.stride = window_size // 2
        self.chunk_size = chunk_size
        # SE splits
        base, rem = divmod(ch, 3)
        self.chs = [base + (1 if i < rem else 0) for i in range(3)]
        self.ses = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(gc, max(gc//reduction,1),1,bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(gc//reduction,1),gc,1,bias=False),
                nn.Sigmoid()
            ) for gc in self.chs
        ])
        # QKV projections
        self.num_heads = num_heads
        self.scale = (ch//num_heads) ** -0.5
        for name in ['to_q_l','to_k_h','to_v_h','to_q_h','to_k_l','to_v_l']:
            setattr(self, name, nn.Conv2d(ch, ch, 1, bias=False))
        self.proj_l = nn.Conv2d(ch, ch, 1, bias=False)
        self.proj_h = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, low, high):
        B,C,H,W = low.shape
        ws,ss = self.window_size, self.stride
        # pad
        n_h = 0 if H<=ws else math.ceil((H-ws)/ss)
        n_w = 0 if W<=ws else math.ceil((W-ws)/ss)
        H_pad,W_pad = n_h*ss+ws, n_w*ss+ws
        pad = (0, W_pad-W, 0, H_pad-H)
        low_pad = F.pad(low, pad)
        high_pad= F.pad(high,pad)
        # SE
        splits = torch.split(high_pad, self.chs, dim=1)
        high_dir = torch.cat([h*se(h) for h,se in zip(splits,self.ses)], dim=1)
        # unfold
        low_unf  = F.unfold(low_pad,  kernel_size=ws, stride=ss)
        high_unf = F.unfold(high_dir,kernel_size=ws, stride=ss)
        _,_,L = low_unf.shape
        out_l = torch.zeros_like(low_unf)
        out_h = torch.zeros_like(high_unf)
        # chunks
        for i in range(0,L,self.chunk_size):
            j = min(i+self.chunk_size, L)
            idx = slice(i,j)
            M = j-i
            lw = low_unf[:,:,idx].transpose(1,2).contiguous().view(-1,C,ws,ws)
            hw = high_unf[:,:,idx].transpose(1,2).contiguous().view(-1,C,ws,ws)
            Bn = lw.size(0); Np=ws*ws
            # QKV
            ql = self.to_q_l(lw).view(Bn,self.num_heads,C//self.num_heads,Np)
            kh = self.to_k_h(hw).view(Bn,self.num_heads,C//self.num_heads,Np)
            vh = self.to_v_h(hw).view(Bn,self.num_heads,C//self.num_heads,Np)
            qh = self.to_q_h(hw).view(Bn,self.num_heads,C//self.num_heads,Np)
            kl = self.to_k_l(lw).view(Bn,self.num_heads,C//self.num_heads,Np)
            vl = self.to_v_l(lw).view(Bn,self.num_heads,C//self.num_heads,Np)
            # attention
            a_lh = torch.softmax(torch.einsum('bhdn,bhdm->bhnm',ql,kh)*self.scale, dim=-1)
            a_hl = torch.softmax(torch.einsum('bhdn,bhdm->bhnm',qh,kl)*self.scale, dim=-1)
            ol = torch.einsum('bhnm,bhdm->bhdn',a_lh,vh).reshape(Bn,C*Np)
            oh = torch.einsum('bhnm,bhdm->bhdn',a_hl,vl).reshape(Bn,C*Np)
            ol = ol.view(B,M,C*Np).transpose(1,2)
            oh = oh.view(B,M,C*Np).transpose(1,2)
            out_l[:,:,idx]=ol; out_h[:,:,idx]=oh
        # fold & norm
        ol_pad = F.fold(out_l,  output_size=(H_pad,W_pad), kernel_size=ws,stride=ss)
        oh_pad = F.fold(out_h,  output_size=(H_pad,W_pad), kernel_size=ws,stride=ss)
        ones   = torch.ones((B,C,H_pad,W_pad), device=low.device)
        mask   = F.fold(F.unfold(ones,kernel_size=ws,stride=ss),output_size=(H_pad,W_pad),kernel_size=ws,stride=ss)
        out_l  = ol_pad/mask; out_h=oh_pad/mask
        return low + self.proj_l(out_l[:,:,:H,:W]), high + self.proj_h(out_h[:,:,:H,:W])

class SwinWindowCrossBand(nn.Module):
    """
    Swin-style shifted-window cross-band attention wrapper
    """
    def __init__(self, ch, window_size=8, reduction=16, num_heads=4, chunk_size=64):
        super().__init__()
        self.win = WindowCrossBandDir(ch, window_size, reduction, num_heads, chunk_size)
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.mlp   = nn.Sequential(nn.Linear(ch,ch*4), nn.GELU(), nn.Linear(ch*4,ch))

    def forward(self, low, high):
        B,C,H,W=low.shape
        # W-MSA
        l1 = self.norm1(low.permute(0,2,3,1)).permute(0,3,1,2)
        h1 = self.norm1(high.permute(0,2,3,1)).permute(0,3,1,2)
        o1_l,o1_h=self.win(l1,h1)
        low1=low+o1_l; high1=high+o1_h
        # SW-MSA
        s= self.win.window_size//2
        ls = torch.roll(low1, shifts=(-s,-s), dims=(2,3))
        hs = torch.roll(high1,shifts=(-s,-s),dims=(2,3))
        ls_n=self.norm2(ls.permute(0,2,3,1)).permute(0,3,1,2)
        hs_n=self.norm2(hs.permute(0,2,3,1)).permute(0,3,1,2)
        o2_l_s,o2_h_s = self.win(ls_n,hs_n)
        o2_l = torch.roll(o2_l_s, shifts=(s,s), dims=(2,3))
        o2_h = torch.roll(o2_h_s, shifts=(s,s), dims=(2,3))
        low2 = low1 + o2_l; high2=high1+o2_h
        # MLP
        l_flat = low2.permute(0,2,3,1).reshape(-1,C)
        h_flat = high2.permute(0,2,3,1).reshape(-1,C)
        ml = self.mlp(l_flat).reshape(B,H,W,C).permute(0,3,1,2)
        mh = self.mlp(h_flat).reshape(B,H,W,C).permute(0,3,1,2)
        return low2+ml, high2+mh


# ---------------------------
# 完整 XNetDeep，window_size 为超参
# ---------------------------
class XNetDeep(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_classes=1,
                 wave='haar', window_size=8, reduction=16, num_heads=4, chunk_size=64):
        super().__init__()
        self.dwt = DWT(wave)
        # encoder level1
        self.lf1 = EncoderBlock(in_channels, base_channels)
        self.hf1 = EncoderBlock(in_channels*3, base_channels)
        self.aspp_l1 = LightASPP(base_channels, base_channels)
        self.aspp_h1 = LightASPP(base_channels, base_channels)
        # encoder level2
        self.lf2 = EncoderBlock(base_channels, base_channels*2)
        self.hf2 = EncoderBlock(base_channels*3, base_channels*2)
        self.aspp_l2 = ASPP(base_channels*2, base_channels*2)
        self.aspp_h2 = ASPP(base_channels*2, base_channels*2)
        # swin-style freq att
        self.freq_att = SwinWindowCrossBand(base_channels, window_size, reduction, num_heads, chunk_size)
        # fusion & decoder & heads
        fused_ch = base_channels + base_channels + 2*base_channels*2
        self.fuse = FusionModule(fused_ch, base_channels*2)
        self.dec1 = DecoderBlock(base_channels*2, base_channels)
        self.dec2 = DecoderBlock(base_channels*3, base_channels)
        self.conv3 = ConvBlock(base_channels*2, base_channels)
        self.conv4 = ConvBlock(base_channels*2, base_channels)
        self.final_conv = nn.Conv2d(base_channels*2, num_classes,1)
    
        self.trunk_aux = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(16, base_channels), base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, num_classes, 1)
        )

        self.aux1 = nn.Conv2d(base_channels,num_classes,1)
        self.aux2 = nn.Conv2d(base_channels,num_classes,1)
        self.res_gate=nn.Conv2d(2,1,3,padding=1)

        self.alpha=nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.res_gate.bias,-5.0)
        nn.init.kaiming_normal_(self.res_gate.weight,mode='fan_in',nonlinearity='sigmoid')
        self.use_gate=True
        self.use_trunk_aux = True
        self.gate_ds = 4          # gate 计算的下采样倍数
        self.beta_edge = 0.5      # Sobel 细化系数（0~1 之间可调）

    def forward(self, x, init_mask):
        # 第1层 DWT & Encoder
        ll1, hf1 = self.dwt(x)
        lf1 = self.lf1(ll1)
        hf1p = self.hf1(hf1)
        a1 = self.aspp_l1(lf1)
        b1 = self.aspp_h1(hf1p)

        # 🚩 将 Attention 从 LF2/HF2 移到 LF1/HF1
        a1_cb, b1_cb = self.freq_att(a1, b1)

        # 低频最大池化下采样
        a1s = F.max_pool2d(a1_cb, 2)
        b1s = F.max_pool2d(b1_cb, 2)

        # 第2层 DWT & Encoder
        ll2, hf2 = self.dwt(lf1)
        lf2 = self.lf2(ll2)
        hf2p = self.hf2(hf2)
        c2 = self.aspp_l2(lf2)
        d2 = self.aspp_h2(hf2p)

        # 🚩 不再对 LF2/HF2 做 Attention，直接用 aspp 输出
        # 统一融合所有特征（特征图大小一致化后）
        c2u = F.interpolate(c2, size=a1s.shape[-2:], mode='bilinear', align_corners=False)
        d2u = F.interpolate(d2, size=a1s.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([a1s, b1s, c2u, d2u], dim=1))

        # Decoder
        d1 = self.dec1(fused)
        lf2u = F.interpolate(lf2, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        d1c = torch.cat([d1, lf2u], dim=1)
        d2_ = self.dec2(d1c)
        aux1 = self.aux1(d2_)
        aux1u = F.interpolate(aux1, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 额外融合 a1_cb（经过 attention）引导细节
        a1u = F.interpolate(a1_cb, size=d2_.shape[-2:], mode='bilinear', align_corners=False)
        d2c = torch.cat([d2_, a1u], dim=1)
        d3 = self.conv3(d2c)
        aux2 = self.aux2(d3)
        aux2u = F.interpolate(aux2, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # 加入 hf1p（经过 attention 的高频特征）
        hf1u = F.interpolate(hf1p, size=d3.shape[-2:], mode='bilinear', align_corners=False)
        d3c = torch.cat([d3, hf1u], dim=1)
        d4 = self.conv4(d3c)

        shared = torch.cat([d3, d4], dim=1) 
        trunk_base = self.final_conv(shared)                   # shape: (B,1,H,W)
        if not self.use_trunk_aux:
            trunk_aux_logits = torch.zeros_like(trunk_base)
        else:
            trunk_aux_logits = self.trunk_aux(shared)
        trunk_pre_sig = torch.sigmoid(trunk_base + trunk_aux_logits)

        # 2) 计算 Gate（或直接给 g=0），都要保证 g 一定被赋值
        if not self.use_gate:
            g = torch.zeros_like(trunk_pre_sig)  # Gate 关闭时直接置零
        else:
            # —— 低分辨率 Gate —— 
            k = self.gate_ds
            trunk_q = F.avg_pool2d(trunk_pre_sig, k, k)          # B×1×(H/4)×(W/4)
            init_q  = F.avg_pool2d(init_mask,     k, k)
            res_q   = init_q - trunk_q
            g_q     = torch.sigmoid(self.res_gate(torch.cat([trunk_q, res_q], 1)))  # B×1×(H/4)×(W/4)

            # 最近邻上采回全尺寸
            g_up = F.interpolate(g_q, size=trunk_pre_sig.shape[-2:], mode='nearest')  # B×1×H×W

            # —— Sobel 边缘细化 —— 
            sobel = kornia.filters.sobel(trunk_pre_sig)                        # B×1×2×H×W
            edge  = torch.sqrt((sobel**2).sum(1, keepdim=True) + 1e-6)          # B×1×H×W
            edge  = (edge - edge.amin(dim=(2,3), keepdim=True)) \
                    / (edge.amax(dim=(2,3), keepdim=True) + 1e-6)
            g = torch.clamp(g_up + self.beta_edge * edge, 0.0, 1.0)            # B×1×H×W

        # 3) 用 g 对 trunk_pre_sig 进行残差修正
        pos_res = F.relu(init_mask - trunk_pre_sig)
        neg_res = F.relu(trunk_pre_sig - init_mask)
        delta   = self.alpha * g * (pos_res - neg_res)    # shape: (B,1,H,W)
        trunk_corrected_logits = (trunk_base + trunk_aux_logits).detach().clone()  # “原始” logit
        # 上一句只是为了示意，下面直接用 logits 叠加：
        logits = (trunk_base + trunk_aux_logits) + delta   # 先把 trunk_aux_logits 加回去，再加上 delta
        main   = torch.sigmoid(logits)

        # 4) 返回顺序： main, aux1u, aux2u, ... , trunk_pre_sig, trunk_aux_logits, g
        return main, aux1u, aux2u, lf1, hf1p, lf2, hf2p, a1_cb, b1_cb, fused, trunk_pre_sig, trunk_aux_logits, g


# --------------------------- Testing ---------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B,C,H,W = 1,3,736,1280
    x = torch.randn(B,C,H,W, device=device)
    init_mask = torch.rand(B,1,H,W, device=device)

    model = XNetDeep(
        in_channels=C,
        base_channels=128,
        num_classes=1,
        wave='haar',
        window_size=16,   # now flexible
        reduction=16,
        num_heads=8,
        chunk_size=64
    ).to(device)

    model.use_gate = False
    with torch.no_grad():
        outs_off = model(x, init_mask)
    print("Gate OFF:", [o.shape for o in outs_off])
    model.use_gate = True; model.alpha.data.fill_(1.0)
    with torch.no_grad():
        outs_on = model(x, init_mask)
    print("Gate ON :", [o.shape for o in outs_on])
