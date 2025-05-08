import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [256, 256]


class ActorHead(nn.Module):
    def __init__(self, token_dim=16):
        super(ActorHead, self).__init__()
        self.fl1 = nn.Linear(token_dim, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], 2)
        self.log_std = nn.Parameter(torch.zeros(2))  # Deviazione standard appresa (globale)

    def forward(self, token, uav_mask):
        """
        token: (B, U, D) - embedding di tutti gli UAV
        uav_mask: (B, U) - booleano, True dove l'UAV è reale
        """
        B, U, D = token.shape

        # Appiattisci le dimensioni batch e UAV
        token_flat = token.view(B * U, D)
        mask_flat = uav_mask.view(B * U)

        # Seleziona solo i token validi
        valid_tokens = token_flat[mask_flat]

        # Applica la rete solo ai token validi
        x = F.relu(self.fl1(valid_tokens))
        x = F.relu(self.fl2(x))
        mean_valid = self.fl3(x)
        std_valid = torch.exp(self.log_std).expand_as(mean_valid)

        # Inizializza i tensori di output
        mean = torch.zeros(B * U, 2, device=token.device)
        std = torch.ones(B * U, 2, device=token.device) * 1e-6

        # Inserisci i valori calcolati solo dove necessario
        mean[mask_flat] = mean_valid
        std[mask_flat] = std_valid

        # Ripristina le dimensioni originali
        mean = mean.view(B, U, 2)
        std = std.view(B, U, 2)
        return mean, std
