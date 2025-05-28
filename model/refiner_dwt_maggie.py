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
            buf = kern.unsqueeze(0).unsqueeze(0)
            self.register_buffer(name, buf)

    def forward(self, x):
        B, C, H, W = x.shape
        filters = {k: v.repeat(C,1,1,1) for k,v in [('ll',self.ll),('lh',self.lh),('hl',self.hl),('hh',self.hh)]}
        LL = F.conv2d(x, filters['ll'], stride=2, groups=C)
        LH = F.conv2d(x, filters['lh'], stride=2, groups=C)
        HL = F.conv2d(x, filters['hl'], stride=2, groups=C)
        HH = F.conv2d(x, filters['hh'], stride=2, groups=C)
        HF = torch.cat([LH,HL,HH], dim=1)
        return LL, HF

# -----------------
# ASPP 模块（含 5×5 并行分支）
# -----------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1,6,12,18), num_groups=16):
        super().__init__()
        self.branches = nn.ModuleList()
        # 空洞卷积分支
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True)
            ))
        # 1x1 卷积分支
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True)
        ))
        # 5x5 大核卷积分支（中尺度信息）
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False),
            nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True)
        ))
        # 全局池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch, affine=True),
            nn.ReLU(inplace=True)
        )
        # 拼接后融合
        self.project = nn.Sequential(
            nn.Conv2d(len(self.branches)*out_ch + out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        res = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        res.append(gp)
        x = torch.cat(res, dim=1)
        return self.project(x)

# -----------------
# 基础模块
# -----------------
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(num_groups,out_ch), out_ch), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(num_groups,out_ch), out_ch), nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x): return self.block(x)

class FusionModule(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.gn   = nn.GroupNorm(min(num_groups,out_ch), out_ch)
        self.act  = nn.LeakyReLU(0.1,inplace=True)
    def forward(self, lf, hf):
        x = torch.cat([lf,hf], dim=1)
        return self.act(self.gn(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=16):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.block  = nn.Sequential(
            nn.GroupNorm(min(num_groups,out_ch), out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(num_groups,out_ch), out_ch), nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.block(self.upconv(x))

# -----------------
# XNetDeep
# -----------------
class XNetDeep(nn.Module):
    """
    多级 DWT + ASPP 深度融合网络（XNet 架构）：
    保留低频和高频分支独立解码，最后拼接(cat)融合。
    上采样使用可学习的 ConvTranspose2d；辅助分支插值用于深度监督。
    """
    def __init__(self, in_channels=3, base_channels=64, num_classes=1, wave='haar'):
        super().__init__()
        self.dwt      = DWT(wave)
        self.lf_enc1  = EncoderBlock(in_channels, base_channels)
        self.hf_enc1  = EncoderBlock(in_channels*3, base_channels)
        self.lf_enc2  = EncoderBlock(base_channels, base_channels*2)
        self.hf_enc2  = EncoderBlock(base_channels*3, base_channels*2)
        self.aspp_lf  = ASPP(base_channels*2, base_channels*2)
        self.aspp_hf  = ASPP(base_channels*2, base_channels*2)
        self.fuse     = FusionModule(base_channels*4, base_channels*2)
        self.lf_dec1  = DecoderBlock(base_channels*2, base_channels)
        self.lf_dec2  = DecoderBlock(base_channels, base_channels)
        self.hf_dec1  = DecoderBlock(base_channels*2, base_channels)
        self.hf_dec2  = DecoderBlock(base_channels, base_channels)
        self.final_conv = nn.Conv2d(base_channels*2, num_classes, 1)
        self.aux1_head  = nn.Conv2d(base_channels, num_classes, 1)
        self.aux2_head  = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        ll1, hf1 = self.dwt(x)
        lf1      = self.lf_enc1(ll1)
        hf1      = self.hf_enc1(hf1)
        ll2, hf2 = self.dwt(lf1)
        lf2      = self.lf_enc2(ll2)
        hf2      = self.hf_enc2(hf2)
        lf2      = self.aspp_lf(lf2)
        hf2      = self.aspp_hf(hf2)
        fused    = self.fuse(lf2, hf2)
        lf_x1    = self.lf_dec1(fused)
        lf_x2    = self.lf_dec2(lf_x1)
        hf_x1    = self.hf_dec1(fused)
        hf_x2    = self.hf_dec2(hf_x1)
        main_cat = torch.cat([lf_x2, hf_x2], dim=1)
        main_raw = self.final_conv(main_cat)
        main_up  = torch.sigmoid(main_raw)
        aux1_raw = self.aux1_head(lf_x1)
        aux2_raw = self.aux2_head(hf_x1)
        aux1_up  = torch.sigmoid(F.interpolate(aux1_raw, size=x.shape[-2:], mode='bilinear', align_corners=False))
        aux2_up  = torch.sigmoid(F.interpolate(aux2_raw, size=x.shape[-2:], mode='bilinear', align_corners=False))
        return main_up, lf1, hf1, lf2, hf2, fused, lf_x1, hf_x1, lf_x2, hf_x2, aux1_up, aux2_up

# -----------------
# Quick Test
# -----------------
if __name__ == "__main__":
    B, C, H, W = 2, 3, 736, 1280
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(B, C, H, W).to(device)
    model = XNetDeep(in_channels=3, base_channels=64).to(device)
    outs = model(x)
    print([o.shape for o in outs])


