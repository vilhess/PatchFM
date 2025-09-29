import torch
import torch.optim as optim
import lightning as L
from model.training.loss import MultiQuantileLoss
from model.training.modules import PatchFM

class PatchFMLit(L.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()

        self.model = PatchFM(
            patch_len=model_config.patch_len, 
            d_model=model_config.d_model, 
            n_heads=model_config.n_heads, 
            n_layers_encoder=model_config.n_layers_encoder, 
            dropout=train_config.dropout, 
            quantiles=model_config.quantiles
        )
        self.criterion = MultiQuantileLoss(self.model.quantiles)

        config = {**model_config.__dict__, **train_config.__dict__}
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch

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
    
    def on_train_epoch_end(self):
        torch.save(self.model.state_dict(), self.hparams.ckpt_path)
