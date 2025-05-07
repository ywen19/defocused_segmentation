import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- Utility ----------------------
def make_conv_block(in_channels, out_channels, depth):
    layers = []
    for i in range(depth):
        layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def make_fc_block(in_dim, out_dim, depth):
    layers = []
    for i in range(depth):
        layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# ------------------- Blur-Aware Swin Attention ----------------------
class BlurAwareSwinAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, ff_dim=512, blur_strength=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = dim ** -0.5
        self.blur_strength = blur_strength

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, blur_map=None):
        B, C, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0

        x = x.view(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, C)

        qkv = self.qkv(x).reshape(x.size(0), x.size(1), 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if blur_map is not None:
            blur_down = F.interpolate(blur_map, size=(H, W), mode='bilinear', align_corners=False)
            blur_down = blur_down.view(B, 1, H // ws, ws, W // ws, ws)
            blur_down = blur_down.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, 1)
            blur_mod = 1 + self.blur_strength * blur_down
            blur_mod = blur_mod.expand(-1, -1, ws * ws).permute(0, 2, 1)
            attn = attn * blur_mod.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.size(0), ws * ws, C)
        out = self.proj(out)
        x = x + self.norm1(out)
        x = x + self.norm2(self.ffn(x))
        x = x.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return x

# ------------------- Defocus Blur Estimator ----------------------
class DefocusBlurEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blur_conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.blur_conv(x)

# ------------------- Decoder Module ----------------------
def make_decoder(mode, in_channels, base_channels):
    if mode == 'pixelshuffle':
        return nn.Sequential(
            nn.Conv2d(in_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )
    elif mode == 'deconv':
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )

# ------------------- Full Model ----------------------
class MatteRefiner(nn.Module):
    def __init__(self,
        base_channels=64,
        enc_depth=2,
        contrastive_depth_enc=2,
        contrastive_depth_trans=2,
        trans_heads=4,
        trans_ff_dim=512,
        trans_depth=3,
        window_size=8,
        decoder_mode='bilinear',
        use_blur_estimator=True,
        use_blur_swin=True,
        use_contrastive=True,
    ):
        super().__init__()
        self.use_blur_estimator = use_blur_estimator
        self.use_blur_swin = use_blur_swin
        self.use_contrastive = use_contrastive

        if use_blur_estimator:
            self.blur_encoder = DefocusBlurEstimator()

        self.encoder = nn.Sequential(
            make_conv_block(3, base_channels, enc_depth),
            nn.Conv2d(base_channels, base_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.encoder_skip = nn.Conv2d(base_channels * 4, base_channels * 4, 1)

        if use_blur_swin:
            self.transformer = nn.Sequential(*[
                BlurAwareSwinAttentionBlock(
                    dim=base_channels * 4,
                    num_heads=trans_heads,
                    window_size=window_size,
                    ff_dim=trans_ff_dim,
                    blur_strength=1.0
                ) for _ in range(trans_depth)
            ])
        else:
            self.transformer = nn.Identity()

        self.decoder = make_decoder(decoder_mode, base_channels * 4, base_channels)

        if use_contrastive:
            self.contrastive_head_enc = nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
                nn.Sigmoid()
            )
            self.contrastive_fc_enc = make_fc_block(base_channels, base_channels // 2, contrastive_depth_enc)

            self.contrastive_head_trans = nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
                nn.Sigmoid()
            )
            self.contrastive_fc_trans = make_fc_block(base_channels, base_channels // 2, contrastive_depth_trans)

    def forward(self, rgb, init_mask):
        blur_map = self.blur_encoder(rgb) if self.use_blur_estimator else None
        enc_feat = self.encoder(rgb)

        if self.use_blur_swin:
            x = enc_feat
            for block in self.transformer:
                x = block(x, blur_map=blur_map)
            trans_feat = x
        else:
            trans_feat = enc_feat

        skip_feat = self.encoder_skip(enc_feat)
        trans_feat = trans_feat + skip_feat

        alpha = self.decoder(trans_feat)

        if not self.use_contrastive:
            return alpha, enc_feat, trans_feat

        def get_contrastive_feat(feat, head, fc):
            attn_map = head(feat)
            weighted = feat[:, :attn_map.shape[1]] * attn_map
            pooled = F.adaptive_avg_pool2d(weighted, 1).view(feat.size(0), -1)
            return fc(pooled)

        ctr_enc = get_contrastive_feat(enc_feat, self.contrastive_head_enc, self.contrastive_fc_enc)
        ctr_trans = get_contrastive_feat(trans_feat, self.contrastive_head_trans, self.contrastive_fc_trans)

        return alpha, enc_feat, trans_feat, ctr_enc, ctr_trans