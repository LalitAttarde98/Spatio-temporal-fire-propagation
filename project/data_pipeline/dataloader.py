import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as T
from scipy.ndimage import gaussian_filter, distance_transform_edt

class AugTransform:
    def __init__(self, spatial_size):
        self.spatial_size = spatial_size
        # wind_speed, terrain_slope, temperatures, velocities, vertical_velocity
        # horizontal_velocity,ff_vx, ff_vy, ff_cumsum, diff_temp, diff_vel,
        # firefronts,
        self.noise_levels = torch.tensor([0.00, 0.00, 1.0, 0.1, 0.1,
                                         0.00, 0.1, 0.1, 0.1,
                                         0.1,0.1,0.1,0.1,0.1,
                                        0.00]).view(1, -1, 1, 1)

    def __call__(self, x):
        x = torch.from_numpy(x).to(torch.float32)
        # horizontal flip
        if torch.rand(1) < 0.4:
            x = torch.flip(x, dims=[-1])  
        # horizontal roll
        if torch.rand(1) < 0.4:
            shift = torch.randint(0, x.shape[-1], (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)
        # Vertical roll
        # if torch.rand(1) < 0.5:
        #     shift = torch.randint(-10, 10, (1,)).item()
        #     x = torch.roll(x, shifts=shift, dims=-2)
        # Blur
        if torch.rand(1) < 0.1:
            x[:,:-1] = T.gaussian_blur(x[:,:-1], kernel_size=(3,3))
        # Gaussian noise
        noise = torch.randn_like(x) * self.noise_levels
        x += noise
        return x

class FireDataset(Dataset):

    def __init__(self, data_cfg, transform=None, Enable_validation=False):

        self.data_dir = data_cfg['data_dir']
        csv_path = data_cfg['data_csv'] if not Enable_validation else data_cfg['val_csv']
        self.enable_transform = data_cfg['enable_augmentation']
        self.transform = transform
        self.Enable_validation = Enable_validation
        
        self.data = np.genfromtxt(csv_path, delimiter=',', dtype=None, names=True, encoding='utf-8')
        
        if self.data.ndim == 0:
            self.data = np.array([self.data.item()], dtype=self.data.dtype)

    def __len__(self):
        return len(self.data)
    
    def compute_sdf(self, firefronts):
        sdf = np.zeros_like(firefronts)
        for t in range(firefronts.shape[0]):
            mask = firefronts[t] > 0.5
            if mask.any() and not mask.all():
                dist_out = distance_transform_edt(~mask).astype(np.float32)
                dist_in  = distance_transform_edt( mask).astype(np.float32)
                sdf[t]   = dist_out - dist_in
            elif mask.all():
                sdf[t] = -distance_transform_edt(np.ones_like(mask)).astype(np.float32)
        return np.clip(sdf / 30.0, -1.0, 1.0)

    def compute_curvature(self,phi, eps=1e-6):
        gy, gx = np.gradient(phi, axis=(1, 2))
        g_mag  = np.sqrt(gx**2 + gy**2 + eps)
        nx, ny = gx / g_mag, gy / g_mag
        # curvature = ∂nx/∂x + ∂ny/∂y
        _, dnx_dx = np.gradient(nx, axis=(1, 2))  
        dny_dy, _ = np.gradient(ny, axis=(1, 2))
        kappa = (dnx_dx + dny_dy).astype(np.float32)
        return np.clip(kappa, -1.0, 1.0)

    def firefront_features(self, firefronts):
        # V_normal = - (dphi/dt) * (grad_phi / |grad_phi|^2)
        smooth = gaussian_filter(firefronts.astype(np.float32), sigma=(0, 0.5, 0.5))

        dphi_dt = np.diff(smooth, axis=0, prepend=smooth[0:1])  #smooth[1:] - smooth[:-1]
        gy, gx = np.gradient(smooth, axis=(1, 2))
        grad = gx**2 + gy**2
        
        mask = grad > 1e-6
        
        vx = np.zeros_like(dphi_dt)
        vy = np.zeros_like(dphi_dt)

        vx[mask] = -dphi_dt[mask] * (gx[mask] / grad[mask])
        vy[mask] = -dphi_dt[mask] * (gy[mask] / grad[mask])

        ff_cumsum = np.cumsum(firefronts, axis=0)

        ros_mag   = 0 #np.sqrt(vx**2 + vy**2).astype(np.float32)
        sdf       = 0 #self.compute_sdf(firefronts)
        curvature = 0 #self.compute_curvature(smooth)

        return vx, vy, ff_cumsum, sdf, ros_mag, curvature

    def __getitem__(self, idx):

        sample_meta = self.data[idx]
        
        shape = (sample_meta['Nt'], sample_meta['Nx'], sample_meta['Ny'])
        
        wind_speed = np.full(shape, sample_meta['u'], dtype='float32') # m/sec
        terrain_slope = np.full(shape, sample_meta['alpha'], dtype='float32') * (np.pi / 180.0) 
        
        temp_path = os.path.join(self.data_dir, sample_meta['theta_filename'])
        temperatures = np.fromfile(temp_path, dtype="<f4").reshape(shape).astype('float32') # in Kelvin
        
        vel_path = os.path.join(self.data_dir, sample_meta['ustar_filename']) 
        velocities = np.fromfile(vel_path, dtype="<f4").reshape(shape).astype('float32') 
        
        ff_path = os.path.join(self.data_dir, sample_meta['xi_filename'])
        firefronts = np.fromfile(ff_path, dtype="<f4").reshape(shape).astype('float32')

        # feature engineering
        vertical_velocity = (velocities - (wind_speed * np.cos(terrain_slope))) / np.sin(terrain_slope + 0.0001)
        horizontal_velocity = wind_speed * np.cos(terrain_slope)
        ff_vx, ff_vy, ff_cumsum, sdf, ros_mag, curvature = self.firefront_features(firefronts)
        diff_temp = np.diff(temperatures, axis=0, prepend=temperatures[0:1])
        diff_vel = np.diff(velocities, axis=0, prepend=velocities[0:1])
            
        features = np.stack([
            wind_speed, 
            terrain_slope, 
            temperatures, 
            velocities,
            vertical_velocity, 
            horizontal_velocity,
            ff_cumsum,
            diff_temp, diff_vel,
           # ff_vx,                
           # ff_vy,                
           # sdf,               
           # ros_mag,           
           # curvature,         
            firefronts,
        ], axis=0)
        
        features = features.transpose(1, 0, 2, 3)
        
        if self.enable_transform and self.transform is not None:
            features = self.transform(features)
            
        return features

class FirePropagation(Dataset):
    def __init__(self, fire_dataset, config=None):
        self.fire_dataset = fire_dataset
        self.in_frames = config['in_frames']
        self.out_frames = config['out_frames']
        total_frames = self.in_frames + self.out_frames
        limit, oversampling = config.get('oversampling', [-1, 0])
        if fire_dataset.Enable_validation:
            limit, oversampling = -1, 0
        
        self.indices = []
        for i in range(len(self.fire_dataset)):
            total_sim_time = self.fire_dataset[i].shape[0]
            if total_sim_time >= total_frames:
                for t_start in range(total_sim_time - total_frames + 1):
                    if oversampling > 0 and t_start > limit:
                        for i in range(oversampling):
                            self.indices.append((i, t_start))
                    else:
                        self.indices.append((i, t_start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
 
        sim_idx, t_start = self.indices[idx]

        full_data = self.fire_dataset[sim_idx] 
        
        t_mid = t_start + self.in_frames
        t_end = t_mid + self.out_frames
        
        input_seq = full_data[t_start:t_mid] 
        
        target_seq = full_data[t_mid:t_end, -1:] 
        
        return input_seq, target_seq