import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, loss_weights = [10.0, 1.0, 10.0, 0.1 ], 
                 pos_weight=10):
        super(SegmentLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.loss_weights = loss_weights
        self.mse = nn.MSELoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.BceLoss = nn.BCELoss(reduction='mean')
        self.pos_weight = pos_weight

    def FocalLoss(self, pred, target):
        BCE_loss = self.BceLoss(pred, target)
        pt = torch.exp(-BCE_loss) 
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        F_loss = F_loss * ((target * self.pos_weight) + 1.0)
        return F_loss.mean()
    
    def DiceLoss(self, pred, target):
        
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

    def PosNegBCE(self, ff_preds, ff_target):
        
        positive_mask = ff_target > 0.5
        negative_mask =  ff_target < 0.5

        if positive_mask.any():
            obj_loss = self.BceLoss(ff_preds, ff_target)
        else:
            obj_loss = torch.tensor(0.0, dtype=torch.float32 ,device=ff_preds.device)

        noobj_loss = self.BceLoss(ff_preds[negative_mask], ff_target[negative_mask])
        
        loss = (obj_loss * 1.0) + (noobj_loss * 10.0)  
        return loss

    def compute_1d_boundary(self, firefronts):
        B, T, H, W = firefronts.shape

        row_indices = torch.arange(H, device=firefronts.device, dtype=firefronts.dtype).view(1, 1, H, 1)

        # Use firefronts directly as a soft mask to keep the operation differentiable
        masked_indices = firefronts * row_indices
        
        frontier_indices = torch.max(masked_indices, dim=2)[0]
        normalized_frontier = frontier_indices / float(H - 1)
        
        return normalized_frontier

    def forward(self, predictions, targets):

        ff_target = targets[:, :, 0, :, :]
        ff_preds = predictions[:, :, 0, :, :]     

        #ff_target = self.compute_1d_boundary(targets[:, :, 0, :, :])
        #ff_preds = predictions   
        
        # focalloss = self.FocalLoss(ff_preds, ff_target)
        # diceloss = self.DiceLoss(ff_preds, ff_target)

        #sample_weights = torch.diff(ff_target, axis=1).sum([1,2,3]).abs() + 1.0 
        
        mse = self.mse(ff_preds, ff_target)  #.mean([1,2,3]) * sample_weights

        #loss =  (self.loss_weights[0]*focalloss) + \
        #        (self.loss_weights[1]*diceloss) 
        #return loss.mean()
        return mse.mean()
        #return self.PosNegBCE(ff_preds, ff_target)