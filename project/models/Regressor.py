import torch
import torch.nn as nn
import torch.nn.functional as F
from .ConvLSTM_memory import ConvLSTMCell_SA

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, drop=0.0):
        layers = [
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if drop > 0: layers.append(nn.Dropout2d(drop))
        super().__init__(*layers)

class Encoder(nn.Module):

    def __init__(self, in_channels, hidden_dim,
                 downsample, drop = 0.1):
        super().__init__()
        H_down, W_down = downsample
        self.H_down = H_down
        self.W_down = W_down

        self.spatial = nn.Sequential(
            ConvBnRelu(in_channels, hidden_dim // 2, drop=drop),
            ConvBnRelu(hidden_dim // 2, hidden_dim,
                       k=(4, 3), s=(2, 1), p=(1, 1), drop=drop),   # H → H/2
        )
      
        self.cell = ConvLSTMCell_SA(hidden_dim, hidden_dim, kernel_size=3)

    def forward(self, x: torch.Tensor):

        B, L, C, H, W = x.shape
        flat = x.reshape(B * L, C, H, W)
        feats = self.spatial(flat)  
        feats = feats.view(B, L, -1, self.H_down, self.W_down)

        state = self.cell.init_hidden(B, (self.H_down, self.W_down), x.device)

        for t in range(L):
            f_t = feats[:, t]                      
            state = self.cell(f_t, state)

        last_feat = feats[:, L - 1]
        return last_feat, state

class Decoder(nn.Module):

    def __init__(self, hidden_dim, future_frames,
                 output_channels = 1, drop = 0.1, input_size=(113, 32)):
        super().__init__()
        self.F = future_frames
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        _, self.W_in = input_size

        self.temporal_proj = nn.Conv2d(hidden_dim, hidden_dim * future_frames,
                                       kernel_size=3,padding=1, bias=False)

        self.time_embed = nn.Parameter(
            torch.zeros(future_frames, hidden_dim, 1, 1)
        )
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        self.head = nn.Sequential(
            nn.Upsample(size=self.input_size, mode='bilinear', align_corners=False),
            ConvBnRelu(hidden_dim, hidden_dim // 2, drop=drop),
            ConvBnRelu(hidden_dim // 2, hidden_dim // 4, drop=drop),
            nn.Conv2d(hidden_dim // 4, output_channels, 3, padding=1),
        )

    def forward(self, h):

        B, C, Hd, Wd = h.shape

        x = self.temporal_proj(h)                    
        x = x.view(B, self.F, C, Hd, Wd)

        time_emb = self.time_embed.expand(self.F, C, Hd, Wd)  
        x = x + time_emb.unsqueeze(0)             

        x = x.view(B * self.F, C, Hd, Wd)
        
        preds = self.head(x)                          
        _, _, H_up, W_up = preds.shape
        return preds.view(B, self.F, 1, H_up, W_up)


class DirectRegressor(nn.Module):

    def __init__(
        self,
        inp_channels= 10,
        hidden_dim= 64,
        future_frames= 20,
        input_size= (113, 32),
        drop= 0.1,
    ):
        super().__init__()
        self.future_frames    = future_frames
        self.inp_channels = inp_channels

        H_in, W_in = input_size
        self.H_ = H_in if H_in % 2 == 0 else H_in - 1   #112
        self.W_ = W_in if W_in % 2 == 0 else W_in - 1   # 32
        H_down = self.H_ // 2                       # 56
        W_down = self.W_                            # 32
        self.input_size = input_size   # (113, 32)

        self.encoder = Encoder(inp_channels, hidden_dim,
                                          (H_down, W_down), drop=drop)

        self.decoder = Decoder(hidden_dim, future_frames,
                                         output_channels=1, drop=drop, input_size=input_size)

        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, L, C, H_orig, W_orig = x.shape
           
        x_flat = x.view(B * L, C, H_orig, W_orig)
        x_flat = F.interpolate(x_flat, size=(self.H_, self.W_), mode='bilinear', align_corners=False)
        x = x_flat.view(B, L, C, self.H_, self.W_)

        last_feat, (h, c, m) = self.encoder(x)

        alpha = torch.sigmoid(self.fusion_alpha)
        z = alpha * h + (1 - alpha) * last_feat 

        preds = self.decoder(z)           

        preds = torch.sigmoid(preds)


        return preds   # (B, 20, 1, 113, 32)

