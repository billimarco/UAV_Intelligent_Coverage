import torch.nn as nn


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim=16):
        super(TransformerEncoderDecoder, self).__init__()

        # TRANSFORMER ENCODER-DECODER
        self.embedding_encoder = nn.Linear(2, embed_dim)  # Embedding per il Transformer encoder
        self.embedding_decoder = nn.Linear(4, embed_dim)  # Embedding per il Transformer decoder

        # LayerNorm for normalization of embeddings
        self.layernorm_encoder = nn.LayerNorm(embed_dim)
        self.layernorm_decoder = nn.LayerNorm(embed_dim)

        # LayerNorm for normalization of output
        self.layernorm_output = nn.LayerNorm(embed_dim)

        self.transformer_enocder_decoder = nn.Transformer(d_model=embed_dim, batch_first=True, num_encoder_layers=2, num_decoder_layers=2)

    def forward(self, GU_positions, UAV_info):
        # GU_positions shape: batch * (n, 2), n = current max connected GU
        # UAV_info shape: batch * (m, 4), m = UAV number

        # Embedding delle sequenze di input
        source = self.embedding_encoder(GU_positions)  # shape: batch * (n, embed_dim)
        target = self.embedding_decoder(UAV_info)  # shape: batch * (m, embed_dim)

        # Normalizzazione degli embeddings
        source = self.layernorm_encoder(source)
        target = self.layernorm_decoder(target)

        # RAPPRESENTAZIONE DELLO STATO PER OGNI UAV
        tokens = self.transformer_enocder_decoder(source, target)  # shape: batch * (m, embed_dim)

        # Normalizzazione dell'output
        tokens = self.layernorm_output(tokens)

        return tokens
