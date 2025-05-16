import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [256, 256]


class CriticHead(nn.Module):
    def __init__(self, token_dim=16):
        super(CriticHead, self).__init__()
        self.fc1 = nn.Linear(token_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, tokens, uav_mask):
        """
        tokens: (B, U, D) - embedding di tutti gli UAV
        uav_mask: (B, U) - booleano, True dove l'UAV è reale
        """
        uav_mask = uav_mask.unsqueeze(-1)                        # (B, U, 1)
        tokens_masked = tokens * uav_mask                    # azzera i token fittizi
        summed = tokens_masked.sum(dim=1)                # somma reali: (B, D)
        count = uav_mask.sum(dim=1).clamp(min=1e-6)          # numero UAV reali per batch
        pooled = summed / count                          # media sui token reali → (B, D)

        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)                   # output: (B,)
