import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [256, 256]

class CriticHeadGlobal(nn.Module):
    def __init__(self, token_dim, action_dim=2):
        super().__init__()
        self.token_action_dim = token_dim + action_dim
        self.pre_attn = nn.Linear(self.token_action_dim, token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(token_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, tokens, uav_mask, actions):
        # tokens: (B, U, D), actions: (B, U, A), uav_mask: (B, U)
        token_action_concat = torch.cat([tokens, actions], dim=-1)  # (B, U, D+A)
        token_action = self.pre_attn(token_action_concat)  # (B, U, D)

        key_padding_mask = ~uav_mask  # True to mask out
        attn_out, _ = self.attn(token_action, token_action, token_action, key_padding_mask=key_padding_mask)  # (B, U, D)

        # Maschera gli output e fai media solo sui UAV validi
        masked_attn = attn_out * uav_mask.unsqueeze(-1).float()  # (B, U, D)
        sum_attn = masked_attn.sum(dim=1)
        count = uav_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled = sum_attn / count  # (B, D)

        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # (B,)
    
class CriticHeadInd(nn.Module):
    def __init__(self, token_dim, action_dim=2):
        super().__init__()
        self.token_action_dim = token_dim + action_dim
        self.pre_attn = nn.Linear(self.token_action_dim, token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(token_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, tokens, uav_mask, actions):
        # tokens: (B, U, D), actions: (B, U, A), uav_mask: (B, U)
        token_action_concat = torch.cat([tokens, actions], dim=-1)  # (B, U, D+A)
        token_action = self.pre_attn(token_action_concat)  # (B, U, D)

        key_padding_mask = ~uav_mask  # True to mask out invalid UAVs
        attn_out, _ = self.attn(token_action, token_action, token_action, key_padding_mask=key_padding_mask)  # (B, U, D)

        # Passa attn_out attraverso la rete neurale per ogni UAV separatamente
        x = F.relu(self.fc1(attn_out))   # (B, U, hidden_sizes[0])
        x = F.relu(self.fc2(x))           # (B, U, hidden_sizes[1])
        values = self.fc3(x).squeeze(-1)  # (B, U)

        # Metti a zero i valori degli UAV invalidi (mascherati)
        values = values * uav_mask.float()

        return values  # (B, U) valori individuali per ogni UAV


