

import torch
import torch.nn.functional as F
import cv2
from tabulate import tabulate
import numpy as np



def calculate_kpis(x, y):
    
    predictions = x[:, :, 0, :, :] 
    targets = y[:, :, 0, :, :]      
    
    timesteps = targets.shape[1]
    thresholds = torch.arange(0.1, 1.0, 0.1)

    mse_results = {t: [] for t in range(timesteps)}
    ssim_results = {t: [] for t in range(timesteps)}

    for thresh in thresholds:
        binary_pred = (predictions > thresh.item()).float()
        
        for t in range(timesteps):

            t_mse = ((targets[:, t] - binary_pred[:, t])**2).mean().item()

            t_ssim = cal_ssim(targets[:, t:t+1], binary_pred[:, t:t+1], L=1)
            
            mse_results[t].append(t_mse)
            ssim_results[t].append(t_ssim.item())

    def format_grid(data_dict, title):
        header = ["T-Step"] + [f"Th:{round(th.item(),1)}" for th in thresholds]
        rows = []
        
        for t in range(timesteps):
            rows.append([f"Step {t}"] + [f"{v:.4f}" for v in data_dict[t]])
        
        cols_data = list(zip(*[data_dict[t] for t in range(timesteps)]))
        sum_row = ["Mean"] + [f"{np.mean(c):.4f}" for c in cols_data]
        rows.append(sum_row)
        
        return f"\n--- {title} ---\n" + tabulate(rows, headers=header, tablefmt="grid")

    mse_table = format_grid(mse_results, "MSE KPI Table")
    ssim_table = format_grid(ssim_results, "SSIM KPI Table")
    
    return mse_table + "\n" + ssim_table



def cal_ssim(img1, img2, kernel_size=11, L=255):

    device = img1.device
    _, channel, _, _ = img1.size()

    kernel1D = cv2.getGaussianKernel(kernel_size, 1.5)
    window = kernel1D * kernel1D.T
    window = torch.tensor(window, device=device, dtype=torch.float32
                          ).expand(1, channel, kernel_size, kernel_size).contiguous()
    K = [0.01, 0.03]
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # sigma^2 = E[X^2] - E[X]^2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = num / den
    mssim = ssim_map.mean() 
    
    return mssim
