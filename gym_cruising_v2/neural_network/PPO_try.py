import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOTry(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4):
        super().__init__()

        # CNN per la mappa
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        map_output_size = 64 * 4 * 4  # = 1024

        # Embed UAV
        self.uav_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU()
        )

        # Embed GU
        self.gu_embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )

        # Proiettore per token UAV arricchito
        self.per_uav_mlp = nn.Sequential(
            nn.Linear(map_output_size + 32 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        )

        # ✨ Self-attention tra UAV
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Normalizzazione post-attn (opzionale ma utile)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, map_exploration, uav_input, gu_input, uav_mask=None, gu_mask=None):
        B, U, _ = uav_input.shape

        # Elaborazione mappa
        map_input = map_exploration.unsqueeze(1)  # [B, 1, H, W]
        map_feat = self.map_cnn(map_input)        # [B, 1024]

        # GU embedding + pooling
        gu_embed = self.gu_embed(gu_input)
        if gu_mask is not None:
            gu_embed = gu_embed * gu_mask.unsqueeze(-1).float()
            gu_mask_sum = gu_mask.sum(1, keepdim=True).clamp(min=1e-6)
            gu_feat = gu_embed.sum(1) / gu_mask_sum
        else:
            gu_feat = gu_embed.mean(1)  # [B, F_gu]

        # UAV embedding
        uav_embed = self.uav_embed(uav_input)
        if uav_mask is not None:
            uav_embed = uav_embed * uav_mask.unsqueeze(-1).float()

        # Concatenazione di contesto per ogni UAV
        map_feat_exp = map_feat.unsqueeze(1).expand(B, U, -1)
        gu_feat_exp = gu_feat.unsqueeze(1).expand(B, U, -1)
        token_input = torch.cat([map_feat_exp, uav_embed, gu_feat_exp], dim=-1)

        # Token iniziali per ogni UAV
        uav_tokens = self.per_uav_mlp(token_input)  # [B, U, embed_dim]

        # ✨ Self-attention tra UAV
        attn_output, _ = self.self_attn(uav_tokens, uav_tokens, uav_tokens, key_padding_mask=~uav_mask if uav_mask is not None else None)

        # Aggiunta residual + norm (stile Transformer)
        uav_tokens = self.ln(uav_tokens + attn_output)

        # Debug
        if torch.isnan(uav_tokens).any():
            print("⚠️ NaN nei token UAV post-attention")

        return uav_tokens  # [B, U, embed_dim]



