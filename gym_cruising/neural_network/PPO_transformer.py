import torch.nn as nn


class PPOTransformer(nn.Module):
    def __init__(self, embed_dim=16):
        super(PPOTransformer, self).__init__()

        # TRANSFORMER ENCODER-DECODER
        self.embedding_encoder = nn.Linear(2, embed_dim)  # Embedding per il Transformer encoder
        self.embedding_decoder = nn.Linear(4, embed_dim)  # Embedding per il Transformer decoder

        # LayerNorm for normalization of embeddings
        self.layernorm_encoder = nn.LayerNorm(embed_dim)
        self.layernorm_decoder = nn.LayerNorm(embed_dim)

        # LayerNorm for normalization of output
        self.layernorm_output = nn.LayerNorm(embed_dim)

        self.transformer_encoder_decoder = nn.Transformer(d_model=embed_dim, batch_first=True, num_encoder_layers=2, num_decoder_layers=2)

    def forward(self, UAV_info, GU_positions, uav_mask=None, gu_mask=None):
        # Embedding
        source = self.embedding_encoder(GU_positions)  # (B, G, D)
        target = self.embedding_decoder(UAV_info)      # (B, U, D)

        # Normalizzazione
        source = self.layernorm_encoder(source)
        target = self.layernorm_decoder(target)

        # Maschere per evitare calcoli su token fittizi
        src_key_padding_mask = ~gu_mask if gu_mask is not None else None  # (B, G)
        tgt_key_padding_mask = ~uav_mask if uav_mask is not None else None  # (B, U)

        # Forward con maschere
        tokens = self.transformer_encoder_decoder(
            src=source,
            tgt=target,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.layernorm_output(tokens)

