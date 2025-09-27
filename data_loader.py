import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sa_conv_lstm import SA_ConvLSTM_Model

from functools import partial

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_path = 'E:/projects/Spatio-temporal/dataset/train.csv'
test_path = 'E:/projects/Spatio-temporal/dataset/test.csv'
train_dir = 'E:/projects/Spatio-temporal/dataset/train/'
test_dir = 'E:/projects/Spatio-temporal/dataset/test/'

class SpatioTemporalDataset(Dataset):
    def __init__(self, dir_path, file_path, mode):

        data = np.genfromtxt(train_path, delimiter=',', dtype=None, names=True, encoding='utf-8')
        self.time_dim, self.x_dim, self.y_dim = data['Nt'], data['Nx'], data['Ny']
        self.mode = mode
        num_samples = len(data)
        self.wind_speeds = [None] * num_samples
        self.terrain_slopes = [None] * num_samples
        self.temperatures = [None] * num_samples
        self.velocities = [None] * num_samples
        self.firefronts = [None] * num_samples
        self.ids = [None] * num_samples

        for index in range(len(data)):
            shape = (self.time_dim[index], self.x_dim[index], self.y_dim[index])
            
            self.wind_speeds[index] = np.full(shape, data['u'][index], dtype='float32')/50.0
            self.terrain_slopes[index] = np.full(shape, data['alpha'][index], dtype='float32')/50.0
            self.temperatures[index] = np.fromfile(os.path.join(dir_path,\
                data['theta_filename'][index]), dtype="<f4").reshape(shape).astype('float32')/100.0
            self.velocities[index] = np.fromfile(os.path.join(dir_path,\
                data['theta_filename'][index]), dtype="<f4").reshape(shape).astype('float32')/50.0
            self.firefronts[index] = np.fromfile(os.path.join(dir_path,\
                data['xi_filename'][index]), dtype="<f4").reshape(shape).astype('float32')
            self.ids[index] = data['id'][index]

    def __len__(self):
        return len(self.time_dim)
    
    def __getitem__(self, idx):
        wind_speed = self.wind_speeds[idx]
        terrain_slope = self.terrain_slopes[idx]
        temp_map = self.temperatures[idx]
        velocity_field = self.velocities[idx]
        firefront = self.firefronts[idx]
        unique_id = self.ids[idx]

        features = np.stack([wind_speed, terrain_slope, temp_map, velocity_field, firefront], axis = 0)
        features = features.transpose(1,0,2,3)
        if self.mode == 'train':
            return features
        elif self.mode == 'test':
            return unique_id, features

train_dataset = SpatioTemporalDataset(train_dir, train_path, 'train')
#test_dataset = SpatioTemporalDataset(test_dir, test_path, 'test')
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True)
#test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

################################################################
# fourier layer
################################################################
class SA_Memory_Module(nn.Module): #SAM 
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
  
        self.layer_qh = nn.Conv2d(input_dim, hidden_dim ,1)
        self.layer_kh = nn.Conv2d(input_dim, hidden_dim,1)
        self.layer_vh = nn.Conv2d(input_dim, hidden_dim, 1)
        
        self.layer_km = nn.Conv2d(input_dim, hidden_dim,1)
        self.layer_vm = nn.Conv2d(input_dim, hidden_dim, 1)
        
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
    def forward(self, h, m):
        batch_size, channel, H, W = h.shape

        K_h = self.layer_kh(h)
        Q_h = self.layer_qh(h)
        V_h = self.layer_vh(h)
        
        K_h = K_h.view(batch_size, self.hidden_dim, H*W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H*W).transpose(1,2)
        V_h = V_h.view(batch_size, self.hidden_dim, H*W)
        
        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim = -1) #batch_size, H*W, H*W
        Z_h = torch.matmul(A_h, V_h.permute(0,2,1)) 

        K_m = self.layer_km(m)
        V_m = self.layer_vm(m)
        
        K_m = K_m.view(batch_size, self.hidden_dim, H*W)
        V_m = V_m.view(batch_size, self.hidden_dim, H*W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim = -1)
        Z_m = torch.matmul(A_m, V_m.permute(0,2,1))
        
        Z_h = Z_h.transpose(1,2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1,2).view(batch_size, self.input_dim, H, W)

        W_z = torch.cat([Z_h , Z_m], dim = 1)
        Z = self.layer_z(W_z)
        
        ## Memory Updating
        combined = self.layer_m(torch.cat([Z, h], dim = 1))
        mo, mg, mi = torch.chunk(combined, chunks=3, dim = 1)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m 

        return new_h, new_m 


class SA_Convlstm_cell(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        #hyperparrams 
        self.input_channels = input_dim
        self.hidden_dim = hid_dim
        self.kernel_size= 3
        self.padding = 1
        self.attention_layer = SA_Memory_Module(hid_dim, hid_dim)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels = self.input_channels + self.hidden_dim, out_channels = 4 * self.hidden_dim, kernel_size=self.kernel_size, padding = self.padding),
            nn.GroupNorm(4* self.hidden_dim, 4* self.hidden_dim ))    

    def forward(self, x, hidden):
        c, h, m = hidden
        combined = torch.cat([x, h], dim = 1)
        combined_conv = self.conv2d(combined)
        i, f, g, o = torch.chunk(combined_conv, 4, dim =1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = torch.mul(f,c)+ torch.mul(i,g)
        h_next = torch.mul(o, torch.tanh(c_next))
        
        # Self-Attention
        h_next, m_next = self.attention_layer(h_next, m)
        
        return h_next, (c_next, h_next, m_next)
    

class Net_Model(nn.Module):  # self-attention convlstm for spatiotemporal prediction model
    def __init__(self,        
        batch_size = 2,
        img_size = (113, 32),
        n_layers = 1,
        input_dim = 5,
        hidden_dim = 16,):
        super(Net_Model, self).__init__()
        # hyperparams
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
            self.cells.append(SA_Convlstm_cell(input_dim, hidden_dim))
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
                                    device=modes.device)

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
        # batch-size, in_channels, x, y...
        x_syms = list(self.einsum_symbols[:order])

        # in_channels, out_channels, x, y...
        weight_syms = list(x_syms[1:])  # no batch-size

        # batch-size, out_channels, x, y...

        weight_syms.insert(1, self.einsum_symbols[order])  # outputs
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

# class Net2d(nn.Module):
#     def __init__(self, modes, width):
#         super(Net2d, self).__init__()

#         self.fourier1 = SimpleBlock2d(modes, modes, width, out_channels = 3)
#         self.fourier2 = SimpleBlock2d(modes, modes, width, out_channels = 3)
#         self.ConvLSTM = SA_ConvLSTM_Model(
#         batch_size = 2,
#         img_size = (113, 32),
#         n_layers = 1,
#         input_dim = 5,
#         hidden_dim = 16,
#         )

#     def forward(self, x):
#         batch_size, time_steps, features, X_dim, Y_dim = x.shape
#        # 
#         #x = x.view(batch_size*time_steps, features, X_dim, Y_dim)
#         #x = self.fourier1(x)

#         x = self.ConvLSTM(x)
#         #x = x.view(batch_size, time_steps, X_dim, Y_dim)
#         x = self.fourier2(x)
#         return x

#####################################################3
seq_len = torch.tensor(5).to(device)
modes = torch.tensor(16).to(device)
width = torch.tensor(3).to(device)

model = Net_Model(batch_size = 2,
        img_size = (113, 32),
        n_layers = 1,
        input_dim = 5,
        hidden_dim = 16,).to(device)

class TotalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1):
        super(TotalLoss, self).__init__()
        self.alpha = 0.25
        self.gamma = 2
        self.smooth = 1e-6
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        self.BceLoss = nn.BCEWithLogitsLoss()

    def FocalLoss(self, pred, target):
        BCE_loss = self.BceLoss(pred, target)
        pt = torch.exp(-BCE_loss) 
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
    def DiceLoss(self, pred, target):
        pred = self.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice
    
    def forward(self, inputs, targets):
        focalloss = self.FocalLoss(inputs[:,-1], targets[:,-1])
        diceloss = self.DiceLoss(inputs[:,-1], targets[:,-1])
        mse1 = self.mse(self.sigmoid(inputs[:,-1]), targets[:,-1])
        mse2 = self.mse(inputs[:,:-1], targets[:,:-1])
        #entropy = self.BceLoss(inputs[:,:-1], self.sigmoid(targets[:,:-1]))
        #print(mse1, mse2, entropy)
        return focalloss + (diceloss*0.1) + (mse2*0.1) + mse1


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler(device)
criterion = TotalLoss(alpha=0.25, gamma=2)
num_epochs = 20


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for idx, features in enumerate(train_dataloader):
        features = features.to(device)
        total_time = features.shape[1]
            
        for time_step in range(total_time-seq_len):
            sequence = features[:,time_step:time_step+seq_len]
            target = features[:,time_step+seq_len,2:]
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device, dtype=torch.float32):
                
                output = model(sequence)
                loss = criterion(output, target) 
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
                
    print(epoch, total_loss)
