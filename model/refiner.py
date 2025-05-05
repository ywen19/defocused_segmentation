import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- Haar DWT ----------------------
class HaarDWT(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh = torch.tensor([[-0.5, 0.5], [0.5, -0.5]], dtype=torch.float32)
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filt', filt)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        out = F.conv2d(x, self.filt, stride=2)
        out = out.view(B, C, 4, H // 2, W // 2)
        out = out.permute(2, 0, 1, 3, 4)
        return out[0], out[1], out[2], out[3]

# ------------------- Wavelet Encoder ----------------------
class WaveletUNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.dwt = HaarDWT()
        self.low_path = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.high_path = nn.Sequential(
            nn.Conv2d(in_channels * 3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Conv2d(base_channels * 3, base_channels * 4, 1)

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        low_feat = self.low_path(ll)
        high_input = torch.cat([lh, hl, hh], dim=1)
        high_feat = self.high_path(high_input)
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([low_feat, high_feat], dim=1)
        return self.fuse(out)

# ------------------- Defocus Blur Estimator ----------------------
class DefocusBlurEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blur_conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.blur_conv(x)

# ------------------- Shallow Attention Encoder ----------------------
class ShallowAttentionEncoder(nn.Module):
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.attn_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x, blur_map=None):
        feat = self.conv1(x)
        attn = torch.sigmoid(self.attn_conv(feat))
        if blur_map is not None:
            blur_map = F.interpolate(blur_map, size=attn.shape[2:], mode='bilinear', align_corners=False)
            attn = attn * (1 + blur_map)
        return feat, attn

# ------------------- Transformer Block ----------------------
class WindowedBlurTransformerBlock(nn.Module):
    def __init__(self, in_dim, window_size=8, heads=4, ff_dim=512):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(in_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, in_dim)
        )
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)

    def forward(self, x, blur_map=None):
        B, C, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, f"Input must be divisible by window size {ws}, but got ({H}, {W})"

        x = x.view(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)

        if blur_map is not None:
            blur_resized = F.interpolate(blur_map, size=(H, W), mode='bilinear', align_corners=False)
            blur = blur_resized.view(B, 1, H // ws, ws, W // ws, ws)
            blur = blur.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, 1)

        attn_out, _ = self.attn(x, x, x)
        if blur_map is not None:
            attn_out = attn_out * (1 + blur)

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))

        x = x.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return x

# ------------------- Decoder ----------------------
class LightEdgePreservingDecoder(nn.Module):
    def __init__(self, in_channels=256, skip_channels=32, mid_channels=64, out_channels=32):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.fuse_skip = nn.Conv2d(mid_channels + skip_channels, mid_channels, 3, padding=1)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.edge_mod = nn.Sequential(
            nn.Conv2d(out_channels, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, attention=None, skip_feat=None):
        x = self.reduce(x)
        x = self.up1(x)
        if skip_feat is not None:
            skip_feat = F.interpolate(skip_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_feat], dim=1)
            x = self.fuse_skip(x)
        x = self.up2(x)

        edge = self.edge_mod(x)
        x = x * (1 + edge)

        if attention is not None:
            att = F.interpolate(attention, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x * (1 + att)

        return torch.sigmoid(self.out_conv(x))

# ------------------- Full Refiner ----------------------
class MatteRefiner(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.base_channels = base_channels
        self.blur_encoder = DefocusBlurEstimator()
        self.shallow = ShallowAttentionEncoder(in_channels=4, base_channels=base_channels // 2)
        self.encoder = WaveletUNetEncoder(in_channels=3, base_channels=base_channels)

        encoder_out_channels = base_channels * 4
        self.transformer = WindowedBlurTransformerBlock(in_dim=encoder_out_channels)

        self.decoder = LightEdgePreservingDecoder(
            in_channels=encoder_out_channels,
            skip_channels=base_channels // 2,
            mid_channels=base_channels,
            out_channels=base_channels // 2
        )

    def forward(self, rgb, m1):
        blur_map = self.blur_encoder(rgb)
        x = torch.cat([rgb, m1], dim=1)
        shallow_feat, attn = self.shallow(x, blur_map=blur_map)
        enc_feat = self.encoder(rgb)
        trans_feat = self.transformer(enc_feat, blur_map=blur_map)
        alpha = self.decoder(trans_feat, attention=attn, skip_feat=shallow_feat)
        return alpha, attn
