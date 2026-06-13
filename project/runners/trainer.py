import torch.optim as optim
import lightning as pl
import torch.nn.functional as F

class FireTrainer(pl.LightningModule):

    def __init__(self, model, loss=None, optim_parameters={'learning_rate':1e-4}):
        super().__init__()
        self.model = model
        self.learning_rate = optim_parameters['learning_rate']
        self.criterion = loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        batch_size = y.size(0)

        logits = self(x)
        loss = self.criterion(logits, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch):

        x, y = batch
        logits = self(x)

        #loss = F.mse_loss(logits, y)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate) 

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, eta_min=1e-6, T_mult=1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

