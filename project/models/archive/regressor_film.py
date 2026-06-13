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
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(hidden_dim // 4, 4), 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.film_proj = nn.Conv2d(hidden_dim, hidden_dim * 2, 1, bias=True)
        
        self.cell = ConvLSTMCell_SA(hidden_dim, hidden_dim, kernel_size=3)

    def forward(self, x: torch.Tensor, static_cond: torch.Tensor):

        B, L, C, H, W = x.shape
        flat = x.reshape(B * L, C, H, W)
        feats = self.spatial(flat)  
        feats = feats.view(B, L, -1, self.H_down, self.W_down)

        state = self.cell.init_hidden(B, (self.H_down, self.W_down), x.device)

        for t in range(L):
            f_t = feats[:, t]                      
            gamma, beta = self.film_proj(static_cond).chunk(2, dim=1)
            f_t = f_t * (1 + gamma) + beta      
            f_t = f_t * self.gate(f_t)   
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
                                       kernel_size=1, bias=False)

        self.time_embed = nn.Parameter(
            torch.zeros(future_frames, hidden_dim, 1, 1)
        )
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        self.film_proj = nn.Conv2d(hidden_dim, hidden_dim * 2, 1, bias=True)

        self.head = nn.Sequential(
            nn.Upsample(size=self.input_size, mode='bilinear', align_corners=False),
            ConvBnRelu(hidden_dim, hidden_dim // 2, drop=drop),
            ConvBnRelu(hidden_dim // 2, hidden_dim // 4, drop=drop),
            nn.Conv2d(hidden_dim // 4, output_channels, 3, padding=1),
        )

        # self.head_1D = nn.Sequential(
        #     ConvBnRelu(hidden_dim, hidden_dim // 2, drop=drop),
        #     nn.AdaptiveMaxPool2d((1, self.W_in)), 
        #     ConvBnRelu(hidden_dim // 2, hidden_dim // 4, drop=drop),
        #     nn.Conv2d(hidden_dim // 4, 1, kernel_size=(1, 3), padding=(0, 1)),
        # )

    def forward(self, h, static_cond):

        B, C, Hd, Wd = h.shape

        x = self.temporal_proj(h)                    
        x = x.view(B, self.F, C, Hd, Wd)

        time_emb = self.time_embed.expand(self.F, C, Hd, Wd)  
        x = x + time_emb.unsqueeze(0)             

        x = x.view(B * self.F, C, Hd, Wd)

        cond = static_cond.unsqueeze(1).expand(B, self.F, C, *static_cond.shape[-2:])
        cond = cond.reshape(B * self.F, C, *static_cond.shape[-2:])

        gamma, beta = self.film_proj(cond).chunk(2, dim=1)
        x = x * (1 + gamma) + beta   
        
        preds = self.head(x)                          
        _, _, H_up, W_up = preds.shape
        return preds.view(B, self.F, 1, H_up, W_up)
    
        # preds = self.head_1D(x)                          
        # _, _, _, W_up = preds.shape
        # return preds.view(B, self.F, W_up)


class DirectRegressor(nn.Module):

    def __init__(
        self,
        static_channels= 2,
        dynamic_channels= 7,
        hidden_dim= 64,
        future_frames= 20,
        input_size= (113, 32),
        drop= 0.1,
    ):
        super().__init__()
        self.future_frames    = future_frames
        self.dynamic_channels = dynamic_channels
        self.static_channels  = static_channels

        H_in, W_in = input_size
        self.H_ = H_in if H_in % 2 == 0 else H_in - 1   #112
        self.W_ = W_in if W_in % 2 == 0 else W_in - 1   # 32
        H_down = self.H_ // 2                       # 56
        W_down = self.W_                            # 32
        self.input_size = input_size   # (113, 32)


        self.static_enc = ConvBnRelu(static_channels, hidden_dim, drop=drop)

        self.static_down = nn.Conv2d(hidden_dim, hidden_dim,
                                     kernel_size=(4, 3), stride=(2, 1),
                                     padding=(1, 1), bias=False)

        self.encoder = Encoder(dynamic_channels, hidden_dim,
                                          (H_down, W_down), drop=drop)

        self.decoder = Decoder(hidden_dim, future_frames,
                                         output_channels=1, drop=drop, input_size=input_size)

        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, L, C, H_orig, W_orig = x.shape

        last_frame = x[:, L - 1, -1:, :, :]               
        x_flat = x.view(B * L, C, H_orig, W_orig)
        x_flat = F.interpolate(x_flat, size=(self.H_, self.W_), mode='bilinear', align_corners=False)
        x = x_flat.view(B, L, C, self.H_, self.W_)

        x_static  = x[:, 0, :self.static_channels] 
        x_dynamic = x[:, :, self.static_channels:]

        static_feat = self.static_enc(x_static)        
        static_feat = self.static_down(static_feat)     

        last_feat, (h, c, m) = self.encoder(x_dynamic, static_feat)

        alpha = torch.sigmoid(self.fusion_alpha)
        z = alpha * h + (1 - alpha) * last_feat + static_feat 

        preds = self.decoder(z, static_feat)           

        preds = torch.sigmoid(last_frame.unsqueeze(1) + preds)
        #preds = torch.sigmoid(preds)


        return preds   # (B, 20, 1, 113, 32)


