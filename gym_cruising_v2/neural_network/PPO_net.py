import torch.nn as nn
import torch
from typing import Tuple
from gym_cruising_v2.neural_network.PPO_transformer import PPOTransformer
from gym_cruising_v2.neural_network.PPO_actor_net import ActorHead
from gym_cruising_v2.neural_network.PPO_critic_net import CriticHeadGlobal, CriticHeadInd
from torch.distributions import Normal

class PPONet(nn.Module):
    def __init__(self, embed_dim: int, max_uav_number: int, map_size: Tuple[int, int], patch_size: int, global_value: bool):
        super().__init__()
        
        self.backbone = PPOTransformer(embed_dim, max_uav_number, map_size, patch_size)
        
        # Actor: Politica (output probabilità di azioni)
        self.actor_head = ActorHead(embed_dim)
        
        # Critic: Funzione di valore
        if global_value:
            self.critic_head = CriticHeadGlobal(embed_dim)
        else:
            self.critic_head = CriticHeadInd(embed_dim)

    def forward(self, map_exploration, uav_input, gu_input, uav_flags, uav_mask=None, gu_mask=None, actions=None):
        """
        Inoltra i dati nella rete PPO completa: backbone, actor, critic.

        Args:
            map_exploration: (B, H, W) - Mappa di esplorazione
            uav_input: (B, U, uav_dim)
            gu_input: (B, G, gu_dim)
            uav_flags: (B, U)
            uav_mask: (B, U) - True dove l'UAV è attivo
            gu_mask: (B, G)
            actions: opzionale (B, U, A)

        Returns:
            actions: (B, U, A)
            log_probs: (B, U)
            entropy: (B, U)
            values: (B,) oppure (B, U)
        """
        
        # Gestione del caso in cui non ci sono GUs (gu_input vuoto) 
        uav_tokens = self.backbone(map_exploration, uav_input, gu_input, uav_flags, uav_mask, gu_mask)  # (B, U, D)
        
        mean, std = self.actor_head(uav_tokens, uav_mask)  # (B, U, 2)

        # Clamping della std per evitare instabilità numeriche
        std = std.clamp(min=1e-5)

        # Distribuzione normale per campionare le azioni
        dist = Normal(mean, std)
        if actions is None:
            raw_actions = dist.rsample()  # campionamento con reparametrizzazione
            

            # Squash in [-1, 1]
            actions = torch.tanh(raw_actions)

            # Applicare la maschera: forzare le azioni degli UAV fittizi a 0
            actions = actions * uav_mask.unsqueeze(-1).float()  # Dove la maschera è True (UAV reale), lascia le azioni intatte
        else:
            # Invertiamo il tanh: azione è già squashed, la "desquashiamo"
            actions = actions.clamp(min=-1 + 1e-5, max=1 - 1e-5)
            raw_actions = torch.atanh(actions)  # Invertiamo il tanh per ottenere l'azione grezza

            
        # log_probs con correzione di tanh (importante per backprop)
        log_probs = dist.log_prob(raw_actions).sum(-1)  # (B, U)
        log_probs -= torch.log(torch.clamp(1 - actions.pow(2), min=1e-5)).sum(-1)# correzione tanh

        # Applicare la maschera ai log_probs: forzare i log_probs degli UAV fittizi a 0
        log_probs = log_probs * uav_mask.float()  # Dove la maschera è True (UAV reale), lascia i log_probs intatti

        # Entropia
        entropy = dist.entropy().sum(-1)  # (B, U)

        # Applicare la maschera all'entropia: gli UAV fittizi non contribuiscono all'entropia
        entropy = entropy * uav_mask.float()  # Dove la maschera è True (UAV reale), lascia l'entropia intatta

        values = self.critic_head(uav_tokens, uav_mask, actions)  # (B,) if global else (B, U)
        return actions, log_probs, entropy, values
    
    def backbone_forward(self, map_exploration, uav_input, gu_input, uav_flags, uav_mask=None, gu_mask=None):
        """
        Estrae i token UAV dal backbone senza passare da actor/critic.

        Returns:
            uav_tokens: (B, U, D)
        """
        
        # Gestione del caso in cui non ci sono GUs (gu_input vuoto)
        uav_tokens = self.backbone(map_exploration, uav_input, gu_input, uav_flags, uav_mask, gu_mask)  # (B, U, D)
            
        return uav_tokens
    
    def get_action(self, uav_tokens, uav_mask=None, actions=None):
        """
        Genera azioni e statistiche associate dalla testa actor.

        Args:
            uav_tokens: (B, U, D)
            uav_mask: (B, U) - True dove l'UAV è attivo
            actions: opzionale (B, U, A)

        Returns:
            actions, log_probs, entropy
        """
        mean, std = self.actor_head(uav_tokens, uav_mask)  # (B, U, 2)

        # Distribuzione normale per campionare le azioni
        dist = Normal(mean, std)
        if actions is None:
            raw_actions = dist.rsample()  # campionamento con reparametrizzazione
            
            # Squash in [-1, 1]
            actions = torch.tanh(raw_actions)

            # Applicare la maschera: forzare le azioni degli UAV fittizi a 0
            actions = actions * uav_mask.unsqueeze(-1).float()  # Dove la maschera è True (UAV reale), lascia le azioni intatte
        else:
            # Invertiamo il tanh: azione è già squashed, la "desquashiamo"
            raw_actions = torch.atanh(actions)  # Invertiamo il tanh per ottenere l'azione grezza

            
        # log_probs con correzione di tanh (importante per backprop)
        log_probs = dist.log_prob(raw_actions).sum(-1)  # (B, U)
        log_probs -= torch.log(1 - actions.pow(2).clamp(max=1 - 1e-5) + 1e-5).sum(-1) # correzione tanh

        # Applicare la maschera ai log_probs: forzare i log_probs degli UAV fittizi a 0
        log_probs = log_probs * uav_mask.float()  # Dove la maschera è True (UAV reale), lascia i log_probs intatti

        # Entropia
        entropy = dist.entropy().sum(-1)  # (B, U)

        # Applicare la maschera all'entropia: gli UAV fittizi non contribuiscono all'entropia
        entropy = entropy * uav_mask.float()  # Dove la maschera è True (UAV reale), lascia l'entropia intatta

        return actions, log_probs, entropy
    
    def get_value(self, uav_tokens, actions, uav_mask=None):
        """
        Calcola i valori dallo stato corrente.

        Returns:
            values: (B,) se global, altrimenti (B, U)
        """
        values = self.critic_head(uav_tokens, uav_mask)  # (B,)
        return values
    
    def get_action_and_value(self, uav_tokens, uav_mask=None, actions=None):
        """
        Versione congiunta di get_action e get_value.

        Returns:
            actions, log_probs, entropy, values
        """
        mean, std = self.actor_head(uav_tokens, uav_mask)  # (B, U, 2)

        # Clamping della std per evitare instabilità numeriche
        std = std.clamp(min=1e-5)

        # Distribuzione normale per campionare le azioni
        dist = Normal(mean, std)
        if actions is None:
            raw_actions = dist.rsample()  # campionamento con reparametrizzazione
            

            # Squash in [-1, 1]
            actions = torch.tanh(raw_actions)

            # Applicare la maschera: forzare le azioni degli UAV fittizi a 0
            actions = actions * uav_mask.unsqueeze(-1).float()  # Dove la maschera è True (UAV reale), lascia le azioni intatte
        else:
            # Invertiamo il tanh: azione è già squashed, la "desquashiamo"
            raw_actions = torch.atanh(actions)  # Invertiamo il tanh per ottenere l'azione grezza

            
        # log_probs con correzione di tanh (importante per backprop)
        log_probs = dist.log_prob(raw_actions).sum(-1)  # (B, U)
        log_probs -= torch.log(1 - actions.pow(2).clamp(max=1 - 1e-5) + 1e-5).sum(-1) # correzione tanh

        # Applicare la maschera ai log_probs: forzare i log_probs degli UAV fittizi a 0
        log_probs = log_probs * uav_mask.float()  # Dove la maschera è True (UAV reale), lascia i log_probs intatti

        # Entropia
        entropy = dist.entropy().sum(-1)  # (B, U)

        # Applicare la maschera all'entropia: gli UAV fittizi non contribuiscono all'entropia
        entropy = entropy * uav_mask.float()  # Dove la maschera è True (UAV reale), lascia l'entropia intatta

        values = self.critic_head(uav_tokens, uav_mask, actions)  # (B,) if global else (B, U)
        return actions, log_probs, entropy, values