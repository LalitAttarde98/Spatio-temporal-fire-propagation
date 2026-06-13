import torch
import torch.nn as nn
import torch.nn.functional as F
from .ConvLSTM_memory import ConvLSTMCell_SA

class Net_Model(nn.Module): 
    def __init__(self,        
        batch_size = 2,
        img_size = (113, 32),
        n_layers = 1,
        input_dim = 5,
        hidden_dim = 16,):
        super(Net_Model, self).__init__()
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.cells, self.batch_norms = [], []
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        
        self.linear_conv = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=1, stride=1)
        
        for i in range(self.n_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            hidden_dim = self.hidden_dim
            self.cells.append(ConvLSTMCell_SA(input_dim, hidden_dim))
            self.batch_norms.append(nn.LayerNorm((self.hidden_dim, *self.img_size)))  

        self.cells = nn.ModuleList(self.cells)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        
        self.fourier1 = SimpleBlock2d(modes = hidden_dim, width = 4, out_channels = 2)
        self.fourier2 = SimpleBlock2d(modes = hidden_dim//2, width = 1, out_channels = 1)

    def init_hidden(self, batch_size, img_size, device):
        h, w = img_size
        hidden_state = (torch.zeros(batch_size, self.hidden_dim, h, w).to(device=device),
                        torch.zeros(batch_size, self.hidden_dim, h, w).to(device=device),
                        torch.zeros(batch_size, self.hidden_dim, h, w).to(device=device))
        states = [] 
        for i in range(self.n_layers):
            states.append(hidden_state)
        return states 
    

    def forward(self, X, hidden = None):
        if hidden == None:
            device = X.device
            hidden = self.init_hidden(batch_size = self.batch_size, img_size = self.img_size, device=device)
        
        inputs_x = None
        for t in range(X.size(1)):
            inputs_x =X[:, t, :, :, :]
            for i, layer in enumerate(self.cells):
                inputs_x, hidden[i] = layer(inputs_x, hidden[i])
                inputs_x = self.batch_norms[i](inputs_x)
        
        inputs_x = X[:, -1, :, :, :]
        for t in range(X.size(1)):
            for i, layer in enumerate(self.cells):
                inputs_x, hidden[i] = layer(inputs_x, hidden[i])
                inputs_x = self.batch_norms[i](inputs_x)

            inputs_x = self.linear_conv(inputs_x)
        
        temp_velocity = self.fourier1(inputs_x)
        firefronts = self.fourier2(inputs_x)

        #return torch.sigmoid(predict)
        return torch.cat([temp_velocity,firefronts], dim=1)

# Base code from https://github.com/daniwi79/fourier_neural_operator/tree/master
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.std = float(2 / (in_channels + out_channels)**0.5)
        self.weights = torch.normal(0, self.std, 
                                    size=(in_channels, out_channels, modes1, modes2), 
                                    dtype=torch.cfloat, 
                                    device=modes1.device)

        self.einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.slices0 = (
            slice(None),  
            slice(None),  # ............... :,
            slice(self.modes1 // 2),
            slice(self.modes2),
        )
        self.slices1 = (
            slice(None),  
            slice(None), 
            slice(-self.modes1 // 2, None),  
            slice(self.modes2),  
        )
    
    def _contract_dense(self, x, weight):
        order = x.ndim
        x_syms = list(self.einsum_symbols[:order])

        weight_syms = list(x_syms[1:]) 
        weight_syms.insert(1, self.einsum_symbols[order]) 
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

        eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

        return torch.einsum(eq, x, weight)

    def forward(self, x):
        batchsize, channels, height, width = x.shape
        
        x = torch.fft.rfft2(x, norm='forward', dim=(-2, -1))

        out_fft = torch.zeros(
            [batchsize, self.out_channels, height, width // 2 + 1],
            dtype=x.dtype,
            device=x.device,
        )
        out_fft[self.slices0] = self._contract_dense(
            x[self.slices0], self.weights[self.slices1])

        out_fft[self.slices1] = self._contract_dense(
            x[self.slices1], self.weights[self.slices0])

        x = torch.fft.irfft2(
            out_fft, s=(height, width), dim=(-2, -1), norm='forward'
        )

        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes, width, out_channels):
        super(SimpleBlock2d, self).__init__()

        self.modes = modes
        self.width = width
        self.out_channels = out_channels
        self.fc0 = nn.Linear(self.width, self.width)

        self.Spectral_layers = nn.ModuleList([SpectralConv2d_fast(
            self.width, self.width, self.modes, self.modes) for _ in range(4)
        ])
        
        self.conv1d_layers = nn.ModuleList([nn.Conv1d(
            self.width, self.width, 1) for _ in range(4)
        ])
        
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(self.width) for _ in range(4)
        ])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channels)
        self.softplus = nn.Softplus()

    def forward(self, x):
        batchsize, _, size_x, size_y = x.shape
        
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        for i in range(4):
            x1 = self.Spectral_layers[i](x)
            x2 = self.conv1d_layers[i](x.flatten(2)).view(batchsize, self.width, size_x, size_y)
            x = self.bn_layers[i](x1 + x2)
            if i < 3:  
                x = F.relu_(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x