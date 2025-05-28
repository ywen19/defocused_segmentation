import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------
# 可微 DWT 模块
# -----------------
class DWT(nn.Module):
    """
    GPU 可微小波变换 (Haar/db1)。
    输入: x (B, C, H, W)
    输出: LL (B, C, H/2, W/2), HF (B, C*3, H/2, W/2)
    """
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
        B, C, H, W = x.shape
        filters = {k: v.repeat(C,1,1,1) for k, v in [('ll', self.ll), ('lh', self.lh), ('hl', self.hl), ('hh', self.hh)]}
        LL = F.conv2d(x, filters['ll'], stride=2, groups=C)
        LH = F.conv2d(x, filters['lh'], stride=2, groups=C)
        HL = F.conv2d(x, filters['hl'], stride=2, groups=C)
        HH = F.conv2d(x, filters['hh'], stride=2, groups=C)
        HF = torch.cat([LH, HL, HH], dim=1)
        return LL, HF

# -----------------
# ASPP 模块
# -----------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1,6,12,18), num_groups=16):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True)
            )
        )
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False),
                nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True)
            )
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(self.branches)+1)*out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))

# -----------------
# 基础块
# -----------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(num_groups, out_ch), out_ch), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(num_groups, out_ch), out_ch), nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class FusionModule(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.gn   = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.act  = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, lf, hf):
        return self.act(self.gn(self.conv(torch.cat([lf, hf], dim=1))))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.block= nn.Sequential(
            nn.GroupNorm(min(num_groups, out_ch), out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(num_groups, out_ch), out_ch), nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.block(self.up(x))

# -----------------
# XNetDeep + 后门控融合
# -----------------
class XNetDeep(nn.Module):
    """
    带强化残差门控的 XNetDeep。
    在 warmup+freeze 阶段可禁用 gate。
    """
    def __init__(self, in_channels=3, base_channels=64, num_classes=1, wave='haar'):
        super().__init__()
        # DWT + 编/解码模块
        self.dwt       = DWT(wave)
        self.lf1_enc   = EncoderBlock(in_channels, base_channels)
        self.hf1_enc   = EncoderBlock(in_channels*3, base_channels)
        self.lf2_enc   = EncoderBlock(base_channels, base_channels*2)
        self.hf2_enc   = EncoderBlock(base_channels*3, base_channels*2)
        self.aspp_lf   = ASPP(base_channels*2, base_channels*2)
        self.aspp_hf   = ASPP(base_channels*2, base_channels*2)
        self.fuse      = FusionModule(base_channels*4, base_channels*2)
        self.lf_dec1   = DecoderBlock(base_channels*2, base_channels)
        self.lf_dec2   = DecoderBlock(base_channels, base_channels)
        self.hf_dec1   = DecoderBlock(base_channels*2, base_channels)
        self.hf_dec2   = DecoderBlock(base_channels, base_channels)
        self.final_conv= nn.Conv2d(base_channels*2, num_classes, 1)
        self.aux1_head = nn.Conv2d(base_channels, num_classes, 1)
        self.aux2_head = nn.Conv2d(base_channels, num_classes, 1)

        # 强化残差门控
        self.res_gate_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(1))  # 从 0 开始
        nn.init.constant_(self.res_gate_conv.bias, -5.0)
        nn.init.kaiming_normal_(self.res_gate_conv.weight, mode='fan_in', nonlinearity='sigmoid')
        # gate 开关：True 表示启用
        self.use_gate = True

    def forward(self, x, init_mask):
        # DWT + 编码
        ll1, hf1 = self.dwt(x)
        lf1      = self.lf1_enc(ll1)
        hf1      = self.hf1_enc(hf1)
        ll2, hf2 = self.dwt(lf1)
        lf2      = self.lf2_enc(ll2)
        hf2      = self.hf2_enc(hf2)
        # ASPP + 融合
        lf2      = self.aspp_lf(lf2)
        hf2      = self.aspp_hf(hf2)
        fused    = self.fuse(lf2, hf2)
        # 解码
        lf_x1    = self.lf_dec1(fused)
        lf_x2    = self.lf_dec2(lf_x1)
        hf_x1    = self.hf_dec1(fused)
        hf_x2    = self.hf_dec2(hf_x1)
        # 主干输出
        main_cat = torch.cat([lf_x2, hf_x2], dim=1)
        main_raw = self.final_conv(main_cat)
        trunk_up = torch.sigmoid(main_raw)
        # 辅助头
        aux1_up  = torch.sigmoid(F.interpolate(self.aux1_head(lf_x1), size=x.shape[-2:], mode='bilinear', align_corners=False))
        aux2_up  = torch.sigmoid(F.interpolate(self.aux2_head(hf_x1), size=x.shape[-2:], mode='bilinear', align_corners=False))

        if not self.use_gate:
            main_up = trunk_up
            g = torch.zeros_like(trunk_up)
        else:
            # 残差门控
            res = init_mask - trunk_up
            gate_in = torch.cat([trunk_up, res], dim=1)
            g = torch.sigmoid(self.res_gate_conv(gate_in))
            main_up = trunk_up + (self.alpha * g) * res

        return main_up, lf1, hf1, lf2, hf2, fused, lf_x1, hf_x1, lf_x2, hf_x2, aux1_up, aux2_up, trunk_up, g


# Quick Test
if __name__ == "__main__":
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dummy input
    B, C, H, W = 2, 3, 736, 1280
    x = torch.randn(B, C, H, W, device=device)
    init_mask = torch.randint(0, 2, (B, 1, H, W), dtype=torch.float32, device=device)

    model = XNetDeep().to(device)
    model.eval()

    names = ['main_up', 'lf1', 'hf1', 'lf2', 'hf2', 'fused',
             'lf_x1', 'hf_x1', 'lf_x2', 'hf_x2',
             'aux1_up', 'aux2_up', 'trunk_up', 'g']

    # 1) Gate OFF
    model.use_gate = False
    with torch.no_grad():
        out_off = model(x, init_mask)
    print("\n=== Gate OFF Shapes ===")
    for name, tensor in zip(names, out_off):
        print(f"{name:8s}: {tuple(tensor.shape)}")

    # 2) Gate ON (alpha=1 for clarity)
    model.use_gate = True
    model.alpha.data.fill_(1.0)
    with torch.no_grad():
        out_on = model(x, init_mask)
    print("\n=== Gate ON Shapes & g stats ===")
    for name, tensor in zip(names, out_on):
        print(f"{name:8s}: {tuple(tensor.shape)}")
    # 打印 gate 权重范围
    print(f"g min/max: {out_on[-1].min().item():.4f}/{out_on[-1].max().item():.4f}")