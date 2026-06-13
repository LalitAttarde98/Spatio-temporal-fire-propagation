import torch
import torch.nn as nn
import torch.nn.functional as F
from .ConvLSTM_memory import ConvLSTMCell_SA
from .ConvLSTM import ConvLSTMCell

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, downsample, enable_attention):
        super(Encoder, self).__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim * 2, kernel_size=(3,3), padding=(1,1)), # 112 -> 112
            nn.BatchNorm2d(input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 2, input_dim * 4, kernel_size=(3,3), padding=(1,1)), # 112 -> 112
            nn.BatchNorm2d(input_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 4, input_dim * 8, kernel_size=(4,3), stride=(2,1), padding=(1,1)), # 112 -> 56
            nn.BatchNorm2d(input_dim * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 8, input_dim * 16, kernel_size=(3,3), padding=(1,1)), # 56 -> 56
            nn.BatchNorm2d(input_dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 16, hidden_dim,  kernel_size=(3,3), padding=(1,1)), # 56 -> 56
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        if enable_attention:
            self.cell = ConvLSTMCell_SA(hidden_dim, hidden_dim, kernel_size=3)
        else:
            self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)

        self.H_down, self.W_down = downsample

    def forward(self, x, curr_state):

        B, L, _, H, W = x.shape
        x = x.view(B * L, -1, H, W)
        x = self.image_encoder(x)
        x = x.view(B, L, -1, self.H_down, self.W_down)
        
        for t in range(L):
            curr_state = self.cell(x[:, t, :, :, :], curr_state)
        
        output = x[:, L-1, :, :, :]
        return output, curr_state
        

class Decoder(nn.Module):
    def __init__(self, hidden_dim, enable_attention):
        super(Decoder, self).__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3,3), padding=(1,1)), # 112 -> 112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),)
        
        if enable_attention:
            self.cell = ConvLSTMCell_SA(hidden_dim, hidden_dim, kernel_size=3)
        else:
            self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)

    def forward(self, x_t, cached_states):

        x_t = self.in_proj(x_t)
        cached_states = self.cell(x_t, cached_states)

        out = cached_states[0]
        
        return out, cached_states


class Seq2Seq(nn.Module):

    def __init__(self, input_dim, hidden_dim, downsample, output_dim, future_frames, 
                 resize_in = [112, 32], resize_back= [113, 32], enable_attention=False):
        super(Seq2Seq, self).__init__()
        self.resize_in = (input_dim, *resize_in)
        self.future_frames = future_frames
        self.resize_back = resize_back
        
        self.encoder = Encoder(input_dim, hidden_dim, downsample, enable_attention)
        
        self.decoder = Decoder(hidden_dim, enable_attention)
                               
        self.predictor = nn.Sequential(
            nn.Upsample(size=resize_back, mode='bilinear', align_corners=False), # 56 -> 113
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=(3,3), padding=(1,1)), # 113 -> 113
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=(3,3), padding=(1,1)), # 113 -> 113
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, output_dim, kernel_size=(3,3), padding=(1,1)), # 113 -> 113
        )
                      
        self.input_channels = input_dim
        self.H_down, self.W_down = downsample

    def forward(self, x):
        B, L_in, C_in, H, W = x.shape
        device = x.device
        last_frame = x[:, L_in-1, C_in-1:C_in, :, :]
        x = F.interpolate(x, size=self.resize_in,mode="nearest",)
        
        cached_states = self.encoder.cell.init_hidden(B, (self.H_down, self.W_down), device)
  
        output, cached_states = self.encoder(x, cached_states)

        decoder_input = output
        
        preds = []
        
        for t in range(self.future_frames):

            decoder_output, cached_states = self.decoder(decoder_input, cached_states)
            
            decoder_input = decoder_output

            motion = self.predictor(decoder_output)
            next_frame = torch.sigmoid(last_frame + motion)
            last_frame = next_frame

            preds.append(next_frame)

            
        preds = torch.stack(preds, dim=1).squeeze(2)
        
        #preds = F.interpolate(preds, size=self.resize_back,mode="nearest",)
        preds = preds.view(B, self.future_frames, 1, *self.resize_back)
        
        return preds