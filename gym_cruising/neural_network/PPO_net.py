import torch.nn as nn
from gym_cruising.neural_network.transformer_encoder_decoder import TransformerEncoderDecoder
from gym_cruising.neural_network.PPO_actor_net import ActorHead
from gym_cruising.neural_network.PPO_critic_net import CriticHead
from torch.distributions import Normal

class PPONet(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        
        self.backbone = TransformerEncoderDecoder(embed_dim)
        
        # Actor: Politica (output probabilità di azioni)
        self.actor_head = ActorHead(embed_dim)
        
        # Critic: Funzione di valore
        self.critic_head = CriticHead(embed_dim)

    def forward(self, gu_input, uav_input):
        """
        gu_input: (B, G, gu_dim)
        uav_input: (B, U, uav_dim)
        """
        uav_tokens = self.backbone(gu_input, uav_input)  # (B, U, D)
        mean, std = self.actor_head(uav_tokens)  # (B, U, 2)
        dist = Normal(mean, std)
        actions = dist.rsample()  # campionamento con reparametrizzazione

        log_probs = dist.log_prob(actions).sum(-1)  # somma su dim. azione (2D) → (B, U)
        entropy = dist.entropy().sum(-1)  # (B, U)
        values = self.critic_head(uav_tokens).squeeze(-1)  # (B,)
        return actions, log_probs, entropy, values
