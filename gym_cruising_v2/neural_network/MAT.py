import torch.nn as nn
import torch


class MAT(nn.Module):
    def __init__(self, embed_dim=16, max_uav_number=3, img_size=(500, 500), patch_size=50):
        super(MAT, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = ((img_size[0] * img_size[1]) // (patch_size ** 2))

        # Proiezioni di input
        self.uav_proj = nn.Linear(4 + max_uav_number + 4, embed_dim)
        self.gu_proj = nn.Linear(2, embed_dim)
        self.map_patch_proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding per i patch della mappa
        self.map_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Type encoding: GU=0, MAP=1
        self.type_embedding = nn.Embedding(2, embed_dim)

        # Normalizzazioni
        self.norm_uav = nn.LayerNorm(embed_dim)
        self.norm_gu = nn.LayerNorm(embed_dim)
        self.norm_map = nn.LayerNorm(embed_dim, eps=1e-06, elementwise_affine=True)

        # LayerNorm for normalization of output
        self.layernorm_output = nn.LayerNorm(embed_dim)

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
        H, W = map_exploration.shape[-2:]  # Ottieni altezza e larghezza della mappa
        

        # 1. Prepara input embedding
        uav_cat = torch.cat([UAV_info, uav_flags], dim=-1)  # [B, U, D1 + D2]
        uav_tokens = self.norm_uav(self.uav_proj(uav_cat))  # → [B, U, D]

        gu_tokens = self.norm_gu(self.gu_proj(GU_positions))    # [B, G, D]
        
        map_feat = self.map_patch_proj(map_exploration.unsqueeze(1))    # [B, embed_dim, img_size[0]/patch_size, img_size[1]/patch_size]
        map_feat = map_feat.flatten(2).transpose(1, 2)            # [B, N, embed_dim] — N = num_patches
        map_feat += self.map_pos_embedding  # Aggiungi positional encoding
        map_tokens = self.norm_map(map_feat)       # [B, N, D]
        
        # --- Type encoding per l'encoder ---
        N_map = map_tokens.shape[1]
        N_gu= gu_tokens.shape[1]

        type_map = torch.zeros(B, N_map, dtype=torch.long, device=map_tokens.device)   # tipo 0
        type_gu = torch.ones(B, N_gu, dtype=torch.long, device=gu_tokens.device)       # tipo 1

        type_embeddings = torch.cat([
            self.type_embedding(type_map),
            self.type_embedding(type_gu)
        ], dim=1) 

        # --- Concatenate encoder input ---
        encoder_input = torch.cat([map_tokens, gu_tokens], dim=1)            # [B, M+G, D]
        type_embeddings = torch.cat([
            self.type_embedding(type_map),
            self.type_embedding(type_gu)
        ], dim=1)                                                            # [B, M+G, D]
        encoder_input += type_embeddings

        # --- Mask setup
        # transformer vuole True = ignora, False = usa
        # --- Encoder padding mask ---
        if gu_mask is not None:
            enc_gu_mask = ~gu_mask                         # [B, G]
        else:
            enc_gu_mask = torch.zeros(B, N_gu, dtype=torch.bool, device=GU_positions.device)
        enc_map_mask = torch.zeros(B, N_map, dtype=torch.bool, device=map_tokens.device)

        # Concateniamo per formare la mask finale
        src_key_padding_mask = torch.cat([enc_map_mask, enc_gu_mask], dim=1)  # [B, total_enc_tokens]
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

