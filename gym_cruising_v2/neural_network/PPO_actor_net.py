import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [256, 256]


class ActorHead(nn.Module):
    def __init__(self, token_dim):
        super(ActorHead, self).__init__()
        self.fl1 = nn.Linear(token_dim, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3_mean = nn.Linear(hidden_sizes[1], 2)
        self.fl3_logstd = nn.Linear(hidden_sizes[1], 2)


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
        
        mean_valid = self.fl3_mean(x)
        log_std_valid = self.fl3_logstd(x).clamp(-20, 2)
        std_valid = torch.exp(log_std_valid)



        # Inizializza i tensori di output
        mean = torch.zeros(B * U, 2, device=token.device)
        std = torch.ones(B * U, 2, device=token.device) * 1e-6

        # Inserisci i valori calcolati solo dove necessario
        mean[mask_flat] = mean_valid
        std[mask_flat] = std_valid

        # Ripristina le dimensioni originali
        mean = mean.view(B, U, 2)
        std = std.view(B, U, 2)
        
        if torch.isnan(mean).any():
            print("⚠️ NaN detected in mean!")
        if torch.isnan(std).any():
            print("⚠️ NaN detected in std!")

        
        return mean, std
        '''
        B, U, D = token.shape

        # Appiattisci le dimensioni batch e UAV
        token_flat = token.view(B * U, D)
        mask_flat = uav_mask.view(B * U)
        
        # Seleziona solo i token validi
        valid_tokens = token_flat[mask_flat]

        # Applica la rete solo ai token validi
        x = F.relu(self.fl1(valid_tokens))
        x = F.relu(self.fl2(x))
        x = F.tanh(self.fl3(x))
        
        act = torch.zeros(B * U, 2, device=token.device)
        act[mask_flat] = x
        act = act.view(B, U, 2)
        
        return act
        '''
        
        '''
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        hidden_sizes = [256, 256]


        class ActorHead(nn.Module):
            def __init__(self, token_dim=16, action_dim=2):
                super(ActorHead, self).__init__()
                self.action_dim = action_dim
                self.fl1 = nn.Linear(token_dim, hidden_sizes[0])
                self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
                self.fl3_mean = nn.Linear(hidden_sizes[1], action_dim)

                # Log standard deviation globale (condiviso tra tutti gli UAV)
                self.log_std = nn.Parameter(torch.zeros(action_dim))

            def forward(self, token, uav_mask):
                """
                token: (B, U, D) - embedding di tutti gli UAV
                uav_mask: (B, U) - booleano, True dove l'UAV è reale
                """
                B, U, D = token.shape

                token_flat = token.view(B * U, D)
                mask_flat = uav_mask.view(B * U)

                valid_tokens = token_flat[mask_flat]
                x = F.relu(self.fl1(valid_tokens))
                x = F.relu(self.fl2(x))
                mean_valid = self.fl3_mean(x)

                # Espandi lo std globale a tutti gli UAV validi
                std_valid = torch.exp(self.log_std).expand_as(mean_valid)

                # Inizializza tensori output
                mean = torch.zeros(B * U, self.action_dim, device=token.device)
                std = torch.ones(B * U, self.action_dim, device=token.device) * 1e-6

                mean[mask_flat] = mean_valid
                std[mask_flat] = std_valid

                mean = mean.view(B, U, self.action_dim)
                std = std.view(B, U, self.action_dim)

                if torch.isnan(mean).any():
                    print("⚠️ NaN detected in mean!")
                if torch.isnan(std).any():
                    print("⚠️ NaN detected in std!")

                return mean, std
        '''