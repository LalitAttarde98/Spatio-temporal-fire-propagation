import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

def create_animation(predictions, targets, output_file='animation.mp4', threshold=0.8):
    
    predictions = predictions[120, :, 0, :, :] 
    targets = targets[120, :, 0, :, :]      
    preds = predictions.cpu().detach().numpy() #> threshold
    targs = targets.cpu().detach().numpy() #> threshold
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 8))
    
    im1 = ax1.imshow(preds[0], cmap='gray',) #vmin=0, vmax=1)
    im2 = ax2.imshow(targs[0], cmap='gray',) #vmin=0, vmax=1)
    
    ax1.set_title("Prediction")
    ax2.set_title("Target")
    ax1.axis('off')
    ax2.axis('off')

    def update(frame):
        im1.set_data(preds[frame])
        im2.set_data(targs[frame])
        fig.suptitle(f"Frame {frame}: MSE={((preds[frame]-targs[frame])**2).mean():.4f}")
        return [im1, im2]

    ani = animation.FuncAnimation(fig, update, frames=preds.shape[0], interval=50, blit=True)
    
    ani.save(output_file, writer='ffmpeg', fps=2)
    plt.close()
    print("Animation saved.")


def create_inference(collect_results, data, image_shape=(20, 113, 32),folder='logs/', threshold=None):
    
    titles = []
    all_preds = []
    for id in data['id'][:4]:
        sample = data[data['id'] == id]
        frame_data = collect_results[id].reshape(image_shape)
        titles.append(f"u:{sample['u']}, alpha:{sample['alpha']}")
        all_preds.append(frame_data)
        
    num_frames = all_preds[0].shape[0]

    fig, axes = plt.subplots(1, 4, figsize=(10, 6))
    
    ims = []
    for i in range(4):
        im = axes[i].imshow(all_preds[i][0], cmap='gray',)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
        ims.append(im)

    fig.suptitle("Frame 0", fontsize=14)
    plt.tight_layout()

    def update(frame_idx):
        for i in range(4):
            ims[i].set_data(all_preds[i][frame_idx])
            
        fig.suptitle(f"Frame {frame_idx}", fontsize=14)
        
        return ims

    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=num_frames, 
        interval=200,   
        blit=False       
    )

    ani.save(folder+'inference.mp4', writer='ffmpeg', fps=2)
        
    plt.close(fig)
    print("Animation saved.")

def invert_1d_boundary(front, H, diff=False, tau=2.0):
    
    front_ind = front * float(H - 1)
    front_ind = front_ind.unsqueeze(2)
    row_ind = torch.arange(H, device=front.device, dtype=front.dtype).view(1, 1, H, 1)
    
    if diff:
        grid = torch.exp(-tau * (row_ind - front_ind) ** 2)
    else:
        rounded_frontier = torch.round(front_ind).long()
        grid = (row_ind == rounded_frontier).to(front.dtype)
        
    return grid