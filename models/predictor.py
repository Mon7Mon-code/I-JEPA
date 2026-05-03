import torch
import torch.nn as nn
from .encoder import TransformerBlock

class Predictor(nn.Module):
    def __init__(self, embed_dim, predictor_embed_dim, num_patches, num_layers, num_heads, mlp_dim):
        super().__init__()
        self.input = nn.Linear(embed_dim, predictor_embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim))
        self.transform = nn.Sequential(*[TransformerBlock(predictor_embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.output = nn.Linear(predictor_embed_dim, embed_dim)

    def forward(self, x, context_indices, mask_indices):
        y = self.input(x)
        y = y + self.pos_embed[:, context_indices, :] 
        mask_tokens = self.pos_embed[:, mask_indices, :].expand(y.shape[0], -1, -1)
        y = torch.cat([y, mask_tokens], dim=1)
        y = self.norm(self.transform(y))
        y = y[:, -mask_tokens.shape[1]:, :]
        return self.output(y)
    