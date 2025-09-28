import torch
import torch.optim as optim
import lightning as L
from model.training.loss import MultiQuantileLoss
from model.training.modules import PatchFM

class PatchFMLit(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        assert config.epochs >= config.epochs_warmups, "epochs must be greater than epochs_warmups"
        assert config.n_warmups > 0, "n_warmups must be greater than 0"
        assert config.epochs_warmups % config.n_warmups == 0, "number of warmups epochs must be divisible by n_warmups"

        self.model = PatchFM(
            seq_len=config.ws, 
            patch_len=config.patch_len, 
            d_model=config.d_model, 
            n_heads=config.n_heads, 
            n_layers_encoder=config.n_layers_encoder, 
            dropout=config.dropout, 
            quantiles=config.quantiles
        )

        self.criterion = MultiQuantileLoss(self.model.quantiles)
        
        self.ctx = [config.ws // config.n_warmups * i for i in range(1, config.n_warmups + 1)]
        self.n_epochs = config.epochs_warmups // config.n_warmups

        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch

        current_epoch = self.current_epoch
        ctx = self.ctx[current_epoch // self.n_epochs if current_epoch < len(self.ctx) * self.n_epochs else -1]
        x = x[:, -ctx:]

        prediction, x_patch = self.model(x)
        y = y.unsqueeze(1)
        y = torch.cat([x_patch, y], dim=1)    
        loss = self.criterion(prediction, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs, eta_min=0)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
