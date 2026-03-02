# put model classes here
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=6, dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, dim, 4, 4)
    def forward(self, x):
        x = self.conv(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        return x,H,W

class TransformerBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim,dim*4),nn.ReLU(),nn.Linear(dim*4,dim))
        self.norm2 = nn.LayerNorm(dim)
    def forward(self,x):
        a,_ = self.attn(x,x,x)
        x = self.norm1(x+a)
        x = self.norm2(x+self.ff(x))
        return x

class ChangeFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = PatchEmbed()
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(4)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,4),
            nn.ReLU(),
            nn.Conv2d(32,1,1)
        )
    def forward(self,t1,t2):
        x1,H,W = self.embed(t1)
        x2,_,_ = self.embed(t2)
        x = torch.abs(x2-x1)
        for b in self.blocks: x = b(x)
        B,N,C = x.shape
        x = x.transpose(1,2).reshape(B,C,H,W)
        return torch.sigmoid(self.decoder(x))