import torch.nn as nn
import torch


class PPOTransformer(nn.Module):
    def __init__(self, embed_dim=16, max_uav_number=3):
        super(PPOTransformer, self).__init__()
        
        # CNN per la mappa
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # Proiezioni di input
        self.uav_proj = nn.Linear(4 + max_uav_number + 4, embed_dim)
        self.gu_proj = nn.Linear(2, embed_dim)
        self.map_patch_proj = nn.Linear(64, embed_dim)

        # Normalizzazioni
        self.norm_uav = nn.LayerNorm(embed_dim)
        self.norm_gu = nn.LayerNorm(embed_dim)
        self.norm_map = nn.LayerNorm(embed_dim)

        # LayerNorm for normalization of output
        self.layernorm_output = nn.LayerNorm(embed_dim)
        
        
        # Positional Encoding learnable (max 512 token)
        #self.positional_encoding = nn.Parameter(torch.randn(1024, embed_dim))  # [max_len, D]

        # Type embeddings: GU=0, MAP=1
        self.type_embedding = nn.Embedding(3, embed_dim)

        
        self.transformer_encoder_decoder = nn.Transformer(d_model=embed_dim, batch_first=True, num_encoder_layers=2, num_decoder_layers=2)
        
    def forward(self, map_exploration, UAV_info, GU_positions, uav_flags, uav_mask=None, gu_mask=None):
        # Fix gu_mask se contiene righe tutte False
        if gu_mask is not None:
            all_false_rows = ~gu_mask.any(dim=1)  # (B,)
            if all_false_rows.any():
                '''
                false_indices = torch.nonzero(all_false_rows, as_tuple=False).squeeze(1).tolist()
                print(f"⚠️ Righe con gu_mask tutto False: {false_indices}. Forzo gu_mask[:, 0] = True e aggiorno GU_positions.")
                '''
                
                gu_mask = gu_mask.clone()
                GU_positions = GU_positions.clone()

                # Imposta gu_mask[:, 0] = True
                gu_mask[all_false_rows, 0] = True

                # Imposta GU_positions[:, 0, :] a un valore neutro/esplorativo
                # Esempio: zero vector (B, D)
                GU_positions[all_false_rows, 0, :] = torch.randn_like(GU_positions[all_false_rows, 0, :])  
                     
        B, U, _ = UAV_info.shape
        G = GU_positions.shape[1]
        

        # 1. Prepara input embedding
        gu_tokens = self.norm_gu(self.gu_proj(GU_positions))    # [B, G, D]
        uav_cat = torch.cat([UAV_info, uav_flags], dim=-1)  # [B, U, D1 + D2]
        uav_tokens = self.norm_uav(self.uav_proj(uav_cat))  # → [B, U, D]
        
        map_feat = self.map_cnn(map_exploration.unsqueeze(1))    # [B, 64, H', W']
        map_feat = map_feat.flatten(2).transpose(1, 2)            # [B, N, 64] — N = H'*W'
        map_tokens = self.norm_map(self.map_patch_proj(map_feat))       # [B, N, D]
        
        # --- Type encoding per l'encoder ---
        B, N_gu, D = gu_tokens.shape
        B, N_map, D = map_tokens.shape
        B, N_uav, D = uav_tokens.shape

        type_gu = torch.full((B, N_gu), 0, dtype=torch.long, device=gu_tokens.device)    # GU = 0
        type_map = torch.full((B, N_map), 1, dtype=torch.long, device=map_tokens.device) # MAP = 1
        type_uav = torch.full((B, N_uav), 2, dtype=torch.long, device=uav_tokens.device) # UAV = 2

        # --- Concatenate encoder input ---
        encoder_input = torch.cat([gu_tokens, map_tokens, uav_tokens], dim=1)            # [B, G+M+U, D]
        type_tokens = torch.cat([
            self.type_embedding(type_gu),
            self.type_embedding(type_map),
            self.type_embedding(type_uav)
        ], dim=1)                                                                         # [B, G+M+U, D]
        encoder_input += type_tokens

        # --- Mask setup
        # transformer vuole True = ignora, False = usa
        # --- Encoder padding mask ---
        if gu_mask is not None:
            enc_gu_mask = ~gu_mask                         # [B, G]
        else:
            enc_gu_mask = torch.zeros(B, G, dtype=torch.bool, device=GU_positions.device)
        enc_map_mask = torch.zeros(B, N_map, dtype=torch.bool, device=map_tokens.device)
        enc_uav_mask = ~uav_mask if uav_mask is not None else torch.zeros(B, N_uav, dtype=torch.bool, device=uav_tokens.device)

        # Concateniamo per formare la mask finale
        src_key_padding_mask = torch.cat([enc_gu_mask, enc_map_mask, enc_uav_mask], dim=1)  # [B, total_enc_tokens]
        #src_key_padding_mask = ~gu_mask if gu_mask is not None else None  # (B, G)
        tgt_key_padding_mask = ~uav_mask if uav_mask is not None else None  # (B, U)

        final_tokens = self.transformer_encoder_decoder(
            src=encoder_input,
            tgt=uav_tokens,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        
        if torch.isnan(final_tokens).any():
            nan_mask = torch.isnan(final_tokens)
            nan_indices = torch.nonzero(nan_mask, as_tuple=False)

            print("⚠️ Transformer UAVxGU-MAP decoder output contains NaN at the following positions:")
            for idx in nan_indices:
                print(f" -> index: {tuple(idx.tolist())}, value: NaN")
        
        return self.layernorm_output(final_tokens)

