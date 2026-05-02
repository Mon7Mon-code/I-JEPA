import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.mod = nn.Conv2d(3, embed_dim, patch_size, patch_size)
    
    def forward(self, x):
        x = self.mod(x)
        y = x.flatten(2).transpose(1, 2)
        return y

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
      super().__init__()
      self.param = nn.Parameter(torch.zeros(1, num_patches, embed_dim))  
    
    def forward(self, x):
        return x + self.param

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 =  nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim,mlp_dim), nn.GELU(), nn.Linear(mlp_dim, embed_dim))
        
    def forward(self, x):
        y = self.norm1(x)
        y = self.mha(y,y,y)[0]
        z = self.mlp(self.norm2(x+y))
        return x + y + z

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_heads, mlp_dim, num_layers):
        super().__init__()
        self.patch = PatchEmbedding(patch_size, embed_dim)
        self.pos = PositionalEncoding((image_size//patch_size)**2, embed_dim)
        self.transform = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch(x)
        x = self.pos(x)
        x = self.transform(x)
        y = self.norm(x)
        return y
    


