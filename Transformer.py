import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = config.get("transformer_dim")
        input_dim = config.get("input_dim")
        self.input_proj = nn.Sequential(nn.Conv1d(config.get("input_dim"), self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())
        self.window_size = config.get("window_size")
        self.encode_position = config.get("encode_position")

        self.transformer = nn.Transformer(d_model = self.transformer_dim,
                                       nhead = config.get("nhead"),
                                       num_encoder_layers = config.get("num_encoder_layers"),
                                       num_decoder_layers  = config.get("num_decoder_layers"),
                                       dim_feedforward = config.get("dim_feedforward"),
                                       dropout = config.get("transformer_dropout"),
                                       activation = config.get("transformer_activation"))
        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size, 1, self.transformer_dim))
        self.query_embed = nn.Embedding(self.window_size, self.transformer_dim)
        config["output_dim"] = input_dim
        self.ln = nn.LayerNorm(self.transformer_dim)

        self.imu_head = nn.Sequential(
            nn.Conv1d(self.transformer_dim, self.transformer_dim // 4, 1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(self.transformer_dim // 4, input_dim, 1)
        )

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
            # Embed in a high dimensional space and reshape to Transformer's expected shape
            src = self.input_proj(src.transpose(1,2)).permute(2, 0, 1)

            if self.encode_position:
                # Add the position embedding
                src += self.position_embed

            query = self.query_embed.weight.unsqueeze(1).repeat(1, src.shape[1], 1)# Shape S x N x C

            # Transformer pass - gets tensor of shape of S x N x C  and outputs tensor of shape S' x N x C,
            # in our case S=S'
            target = self.ln(self.transformer(src,query)).permute(1, 2, 0)

            return self.imu_head(target).transpose(1, 2)
