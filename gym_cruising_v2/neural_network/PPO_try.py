import torch.nn as nn
import torch


class PPOTry(nn.Module):
    def __init__(self, embed_dim=32, map_shape=(64, 64)):
        super().__init__()
        # CNN per la mappa
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # [B, 16, 250, 250]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # [B, 32, 125, 125]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # [B, 64, 63, 63]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # [B, 64, 4, 4]
            nn.Flatten()                   # → 1024
        )
        map_output_size = 64 * 4 * 4  # = 1024

        # Embed UAV (es. pos, dir, energia, ecc.)
        self.uav_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU()
        )

        # Embed GU (es. posizione utente)
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

    def forward(self, map_exploration, uav_input, gu_input, uav_mask=None, gu_mask=None):
        B, U, _ = uav_input.shape

        # Mappa: [B, H, W] → [B, F_map]
        map_input = map_exploration.unsqueeze(1)
        map_feat = self.map_cnn(map_input)  # [B, F_map]

        # GU embedding → [B, G, F_gu]
        gu_embed = self.gu_embed(gu_input)
        if gu_mask is not None:
            gu_embed = gu_embed * gu_mask.unsqueeze(-1).float()
            gu_feat = gu_embed.sum(1) / gu_mask.sum(1).clamp(min=1e-6).unsqueeze(-1)
        else:
            gu_feat = gu_embed.mean(1)  # [B, F_gu]

        # UAV embedding → [B, U, F_uav]
        uav_embed = self.uav_embed(uav_input)
        if uav_mask is not None:
            uav_embed = uav_embed * uav_mask.unsqueeze(-1).float()

        # Espandi context per ogni UAV
        map_feat_exp = map_feat.unsqueeze(1).expand(B, U, -1)   # [B, U, F_map]
        gu_feat_exp = gu_feat.unsqueeze(1).expand(B, U, -1)     # [B, U, F_gu]

        # Concatenazione per-UAV: [B, U, F_map + F_uav + F_gu]
        token_input = torch.cat([map_feat_exp, uav_embed, gu_feat_exp], dim=-1)

        # Proiezione finale per token UAV → [B, U, embed_dim]
        uav_tokens = self.per_uav_mlp(token_input)

        return uav_tokens  # token per UAV attivo (mascherabili)


