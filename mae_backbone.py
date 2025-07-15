import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, Block, PatchEmbed

# MAEBackbone: Trích encoder từ MAE
class MAEBackbone(nn.Module):
    def __init__(self, mae_encoder: nn.Module):
        super().__init__()
        self.patch_embed = mae_encoder.patch_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, mae_encoder.embed_dim))
        self.pos_embed = mae_encoder.pos_embed
        self.blocks = mae_encoder.blocks
        self.norm = mae_encoder.norm
        self.out_channels = mae_encoder.embed_dim  # để dùng cho FasterRCNN

    def forward(self, x):
        # Resize input nếu chưa đúng size 224x224
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        x = self.patch_embed(x)  # (B, num_patches, dim)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + N, dim)
        x = x + self.pos_embed[:, :x.size(1), :]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x[:, 1:, :]  # bỏ cls token
        H = W = int(x.size(1) ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)  # (B, C, H, W)
        return x

# Hàm build backbone từ checkpoint
def build_mae_backbone(ckpt_path):
    # Khởi tạo model cấu trúc tương ứng (ví dụ MAE-Huge)
    mae_encoder = VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm
    )

    # Load checkpoint từ PyTorch Lightning
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']

    # Chỉ lấy phần encoder
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("mae.encoder."):
            encoder_state_dict[k.replace("mae.encoder.", "")] = v

    # Load weights vào encoder
    mae_encoder.load_state_dict(encoder_state_dict, strict=False)

    # Bọc encoder lại thành backbone dùng cho detection
    backbone = MAEBackbone(mae_encoder)
    return backbone
backbone = build_mae_backbone("/home/tiennv/phucth/medical/mae/outputs_rand_4_bitwise_3_semi_objmask_150_huge/files/output_ptln/sample-epoch=002-valid/loss=0.04.ckpt")
x = torch.randn(2, 3, 224, 224)
out = backbone(x)
print(out.shape)  # Expect: (2, 1280, 14, 14) nếu patch_size=16
