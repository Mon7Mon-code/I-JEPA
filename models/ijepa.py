import torch
import torch.nn as nn
from .encoder import ViT
from .predictor import Predictor

class IJEPA(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_heads, mlp_dim, num_layers, momentum, pred_num_layers):
        super().__init__()
        self.momentum = momentum
        self.context = ViT(image_size, patch_size, embed_dim, num_heads, mlp_dim, num_layers)
        self.target = ViT(image_size, patch_size, embed_dim, num_heads, mlp_dim, num_layers)
        self.target.load_state_dict(self.context.state_dict())
        self.predictor = Predictor(embed_dim, embed_dim, (image_size//patch_size)**2, pred_num_layers, num_heads, mlp_dim)
        for i in self.target.parameters():
            i.requires_grad = False
        
    def update_target(self):
        with torch.no_grad():
            for i, j in zip(self.target.parameters(), self.context.parameters()):
                i.data = self.momentum * i.data + (1 - self.momentum) * j.data
    
    def forward(self, x, cindex, tindex):
        y = self.context(x, cindex)
        with torch.no_grad():
            z = self.target(x)
        a = []
        for t in tindex:
            b = self.predictor(y, cindex, t)
            a.append(b)
        return a,z