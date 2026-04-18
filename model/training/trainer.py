import lightning as L
import torch
import torch.optim as optim
from lightning.pytorch.utilities import grad_norm

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
            quantiles=model_config.quantiles,
            use_xsa=model_config.use_xsa,
        )
        self.criterion = MultiQuantileLoss(self.model.quantiles)

        config = {**model_config.__dict__, **train_config.__dict__}
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x = batch

        # Data augmentation: random sign flip and time flip
        sign_flip = torch.where(
            torch.randn(x.size(0), 1, device=x.device) > 0, 1.0, -1.0
        )
        x = sign_flip * x
        time_flip = torch.randn((x.size(0)), device=x.device) > 0.0
        x[time_flip] = x[time_flip].flip(dims=[1])

        prediction, y = self.model(x)
        prediction = prediction[:, :-1]
        loss = self.criterion(prediction, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):

        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.start_lr, weight_decay=0.01
        )

        div_factor = self.hparams.max_lr / self.hparams.start_lr
        final_div_factor = self.hparams.start_lr / self.hparams.lower_lr
        pct_start = self.hparams.reach_max / self.hparams.iter_cycle
        onecycle = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.iter_cycle,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        constant = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=final_div_factor, total_iters=1e8
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[onecycle, constant],
            milestones=[self.hparams.iter_cycle],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def on_train_epoch_end(self):
        torch.save(self.model.state_dict(), self.hparams.ckpt_path)
