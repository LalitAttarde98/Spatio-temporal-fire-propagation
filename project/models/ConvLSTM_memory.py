import torch
import torch.nn as nn
from .Self_attention import SA_Memory_Module

class ConvLSTMCell_SA(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size):

        super(ConvLSTMCell_SA, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2, kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,)
        
        self.attention_layer = SA_Memory_Module(hidden_dim, hidden_dim)

    def forward(self, input_tensor, cached_state):

        hidden, cell_state, memory = cached_state
        
        combined = torch.cat([input_tensor, hidden], dim=1)  # (B, C_in + C_hid, H, W)

        combined_conv = self.conv(combined) # (B, 4 * C_hid, H, W)
        
        input, forget, output, cell = torch.split(combined_conv, self.hidden_dim, dim=1)

        input_gate = torch.sigmoid(input)  
        forget_gate = torch.sigmoid(forget) 
        output_gate = torch.sigmoid(output)
        cell_gate = torch.tanh(cell)   

        c_next = (forget_gate * cell_state) + (input_gate * cell_gate)
        
        h_next = output_gate * torch.tanh(c_next)
        
        h_next, m_next = self.attention_layer(h_next, memory)

        cached_state = (h_next, c_next, m_next)

        return cached_state

    def init_hidden(self, batch_size, image_size, device):

        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device),)