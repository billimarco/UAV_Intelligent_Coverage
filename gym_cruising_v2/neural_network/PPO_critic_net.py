import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [256, 256]

import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticHead(nn.Module):
    def __init__(self, token_dim=16, action_dim=2):
        super(CriticHead, self).__init__()
        self.token_action_dim = token_dim + action_dim
        self.query = nn.Parameter(torch.randn(1, 1, self.token_action_dim))  # [1, 1, D+A]
        self.attn = nn.MultiheadAttention(embed_dim=self.token_action_dim, num_heads=1, batch_first=True)

        self.fc1 = nn.Linear(self.token_action_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, tokens, uav_mask, actions):
        """
        tokens: (B, U, D)
        actions: (B, U, A)
        uav_mask: (B, U)
        """
        # Concatenazione token + azione per ogni UAV
        token_action = torch.cat([tokens, actions], dim=-1)  # (B, U, D+A)

        # Attention pooling con query appresa
        B = token_action.size(0)
        Q = self.query.expand(B, 1, -1)  # [B, 1, D+A]
        key_padding_mask = ~uav_mask  # True = mask out

        context, _ = self.attn(Q, token_action, token_action, key_padding_mask=key_padding_mask)  # [B, 1, D+A]
        pooled = context.squeeze(1)  # [B, D+A]

        # Passaggio nel Critic
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # [B]
