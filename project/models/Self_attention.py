import torch
import torch.nn as nn
import torch.nn.functional as F

# Base code from https://github.com/tsugumi-sys/SA-ConvLSTM-Pytorch/tree/main/self_attention_memory_convlstm
class SA_Memory_Module(nn.Module): 
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.hidden_q = nn.Conv2d(input_dim, hidden_dim, 1)
        self.hidden_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.hidden_v = nn.Conv2d(input_dim, hidden_dim, 1)

        self.memory_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.memory_v = nn.Conv2d(input_dim, hidden_dim, 1)

        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, h, m):
        B, C, H, W = h.shape

        Q_h = self.hidden_q(h).flatten(2).permute(0, 2, 1)  # (B, HW, hidden_dim)
        K_h = self.hidden_k(h).flatten(2).permute(0, 2, 1) 
        V_h = self.hidden_v(h).flatten(2).permute(0, 2, 1)  

        Z_h = F.scaled_dot_product_attention(Q_h.unsqueeze(1),  # (B, 1, HW, hidden_dim)
                                             K_h.unsqueeze(1),
                                             V_h.unsqueeze(1)).squeeze(1)
        # (B, HW, hidden_dim)
        K_m = self.memory_k(m).flatten(2).permute(0, 2, 1)
        V_m = self.memory_v(m).flatten(2).permute(0, 2, 1)

        Z_m = F.scaled_dot_product_attention(Q_h.unsqueeze(1),
                                             K_m.unsqueeze(1),
                                             V_m.unsqueeze(1)).squeeze(1)

        Z_h = Z_h.permute(0, 2, 1).view(B, self.input_dim, H, W)
        Z_m = Z_m.permute(0, 2, 1).view(B, self.input_dim, H, W)

        Z = self.layer_z(torch.cat([Z_h, Z_m], dim=1))

        # memory update
        combined = self.layer_m(torch.cat([Z, h], dim=1))
        mo, mg, mi = torch.chunk(combined, 3, dim=1)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m