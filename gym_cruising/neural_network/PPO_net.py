import torch.nn as nn
import torch
from gym_cruising.neural_network.PPO_transformer import PPOTransformer
from gym_cruising.neural_network.PPO_actor_net import ActorHead
from gym_cruising.neural_network.PPO_critic_net import CriticHead
from torch_geometric.nn import GCNConv
from torch.distributions import Normal

class PPONet(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        
        self.backbone = PPOTransformer(embed_dim)
        
        # Actor: Politica (output probabilità di azioni)
        self.actor_head = ActorHead(embed_dim)
        
        # Critic: Funzione di valore
        self.critic_head = CriticHead(embed_dim)

    def forward(self, uav_input, gu_input, uav_mask=None, gu_mask=None):
        """
        gu_input: (B, G, gu_dim) → Ground Unit features
        uav_input: (B, U, uav_dim) → UAV features
        """
        
        # Gestione del caso in cui non ci sono GUs (gu_input vuoto)
        if gu_input.size(1) == 0:  # Se il numero di GUs è 0
            uav_tokens = self.backbone(uav_input, gu_input, uav_mask)  # (B, U, D)
        else:
            uav_tokens = self.backbone(uav_input, gu_input, uav_mask, gu_mask)  # (B, U, D)
        
        mean, std = self.actor_head(uav_tokens, uav_mask)  # (B, U, 2)

        # Clamping della std per evitare instabilità numeriche
        std = std.clamp(min=1e-6)

        # Distribuzione normale per campionare le azioni
        dist = Normal(mean, std)
        raw_actions = dist.rsample()  # campionamento con reparametrizzazione

        # Squash in [-1, 1]
        actions = torch.tanh(raw_actions)

        # Applicare la maschera: forzare le azioni degli UAV fittizi a 0
        actions = actions * uav_mask.unsqueeze(-1).float()  # Dove la maschera è True (UAV reale), lascia le azioni intatte

        # log_probs con correzione di tanh (importante per backprop)
        log_probs = dist.log_prob(raw_actions).sum(-1)  # (B, U)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(-1)  # correzione tanh

        # Applicare la maschera ai log_probs: forzare i log_probs degli UAV fittizi a 0
        log_probs = log_probs * uav_mask.float()  # Dove la maschera è True (UAV reale), lascia i log_probs intatti

        # Entropia
        entropy = dist.entropy().sum(-1)  # (B, U)

        # Applicare la maschera all'entropia: gli UAV fittizi non contribuiscono all'entropia
        entropy = entropy * uav_mask.float()  # Dove la maschera è True (UAV reale), lascia l'entropia intatta

        values = self.critic_head(uav_tokens, uav_mask)  # (B,)
        return actions, log_probs, entropy, values
    
    def backbone_forward(self, uav_input, gu_input, uav_mask=None, gu_mask=None):
        """
        gu_input: (B, G, gu_dim) → Ground Unit features
        uav_input: (B, U, uav_dim) → UAV features
        """
        
        uav_tokens = self.backbone(gu_input, uav_input)
        return uav_tokens
    
    def get_action(self, uav_tokens, uav_mask=None):
        """
        gu_input: (B, G, gu_dim) → Ground Unit features
        uav_input: (B, U, uav_dim) → UAV features
        """
        mean, std = self.actor_head(uav_tokens)  # (B, U, 2)
        std = std.clamp(min=1e-6)
        dist = Normal(mean, std)
        raw_actions = dist.rsample()  # campionamento con reparametrizzazione squash in [-1, 1]
        # squash in [-1, 1]
        actions = torch.tanh(raw_actions)
        
        # log_probs con correzione di tanh (importante per backprop)
        log_probs = dist.log_prob(raw_actions).sum(-1)  # (B, U)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(-1)  # correzione tanh  # somma su dim. azione (2D) → (B, U)
        entropy = dist.entropy().sum(-1)  # (B, U)
        return actions, log_probs, entropy
    
    def get_value(self, uav_tokens):
        """
        gu_input: (B, G, gu_dim) → Ground Unit features
        uav_input: (B, U, uav_dim) → UAV features
        """
        values = self.critic_head(uav_tokens)
        return values
    
    def get_action_and_value(self, uav_tokens):
        """
        gu_input: (B, G, gu_dim) → Ground Unit features
        uav_input: (B, U, uav_dim) → UAV features
        """       
        mean, std = self.actor_head(uav_tokens)  # (B, U, 2)
        std = std.clamp(min=1e-6)
        dist = Normal(mean, std)
        raw_actions = dist.rsample()  # campionamento con reparametrizzazione squash in [-1, 1]
        # squash in [-1, 1]
        actions = torch.tanh(raw_actions)
        
        # log_probs con correzione di tanh (importante per backprop)
        log_probs = dist.log_prob(raw_actions).sum(-1)  # (B, U)
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(-1)  # correzione tanh  # somma su dim. azione (2D) → (B, U)
        entropy = dist.entropy().sum(-1)  # (B, U)
        values = self.critic_head(uav_tokens)
        return actions, log_probs, entropy, values
