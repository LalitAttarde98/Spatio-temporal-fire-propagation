import torch
import torch.nn as nn
import torch.nn.functional as F
from .ConvLSTM import ConvLSTMCell

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, downsample, output_channels):
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
            nn.Conv2d(input_dim * 8, input_dim * 16, kernel_size=(4,3), stride=(2,1), padding=(1,1)), # 56 -> 28
            nn.BatchNorm2d(input_dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim * 16, hidden_dim,  kernel_size=(3,3), padding=(1,1)), # 28 -> 28
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)

        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3,3), padding=(1,1)), # 112 -> 112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),)
        
        self.H_down, self.W_down = downsample

    def forward(self, x, curr_state):

        B, L, _, H, W = x.shape
        x = x.view(B * L, -1, H, W)
        x = self.image_encoder(x)
        x = x.view(B, L, -1, self.H_down, self.W_down)
        
        for t in range(L):
            curr_state = self.cell(x[:, t, :, :, :], curr_state)

        output = self.out_proj(curr_state[0])
        
        return output, curr_state
        

class Decoder(nn.Module):
    def __init__(self, hidden_dim, downsample, output_channels):
        super(Decoder, self).__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3,3), padding=(1,1)), # 112 -> 112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),)
        
        self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)

        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3,3), padding=(1,1)), # 112 -> 112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),)

        self.H_down, self.W_down = downsample

    def forward(self, x_t, cached_states):

        x_t = self.in_proj(x_t)
        cached_states = self.cell(x_t, cached_states)

        out = self.out_proj(cached_states[0])
        
        return out, cached_states


class Seq2Seq_sliding(nn.Module):

    def __init__(self, input_dim, hidden_dim, downsample, 
                 output_dim, future_frames, resize_in = [112, 32], resize_back= [113, 32]):
        super(Seq2Seq_sliding, self).__init__()
        self.resize_in = (input_dim, *resize_in)
        self.future_frames = future_frames
        self.resize_back = resize_back
        
        self.encoder = Encoder(input_dim, hidden_dim, downsample, output_dim)
        
        self.decoder = Decoder(hidden_dim, downsample, output_dim)

        self.segment_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3,3), padding=(1,1)), # 112 -> 112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3,3), padding=(1,1)), # 112 -> 112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=(3,3), padding=(1,1)), # 112 -> 112
            nn.Sigmoid(),
        )
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
            )
                      
        self.input_channels = input_dim
        self.H_down, self.W_down = downsample

    def forward(self, x):
        x = F.interpolate(x, size=self.resize_in,mode="nearest",)
        B, L_in, C_in, H, W = x.shape
        device = x.device
        
        cached_states = self.encoder.cell.init_hidden(B, (self.H_down, self.W_down), device)
  
        output, cached_states = self.encoder(x, cached_states)

        decoder_input = output
        
        preds = []
        
        for t in range(self.future_frames):

            decoder_output, cached_states = self.decoder(decoder_input, cached_states)
            preds.append(decoder_output)
            decoder_input = decoder_output
            
        preds = torch.stack(preds, dim=1)
        preds = preds.view(B * self.future_frames, -1, self.H_down, self.W_down)

        patch_pos = self.regression_head(preds)
        patch = self.segment_head(preds)

        predictions = self.differentiable_paste(patch, patch_pos)

        predictions = predictions.view(B, self.future_frames, -1, *self.resize_back)
        patch_pos = patch_pos.view(B, self.future_frames)
        
        return (predictions, patch_pos)

    def differentiable_paste(self, patch, pos_norm):

        N, C, ph, pw = patch.shape
        fh, fw = self.resize_back 
        device = patch.device

        pos_grid = (pos_norm * 2.0) - 1.0

        scale_y = fh / ph
        scale_x = fw / pw
        
        # y_s = scale_y * (y_t - pos_grid)
        ty = -1.0  - (scale_y * pos_grid.squeeze())

        theta = torch.zeros(N, 2, 3, device=device)
        theta[:, 0, 0] = scale_x   
        theta[:, 1, 1] = scale_y   
        theta[:, 1, 2] = ty        

        grid = F.affine_grid(theta, torch.Size((N, C, fh, fw)), align_corners=False)
        full = F.grid_sample(patch, grid, mode='nearest',
                                padding_mode='zeros', align_corners=False)
        
        return full
