import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger

from model import PatchFMLit
from dataset import artificial_dataset

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    seed_everything(0, workers=True)

    OmegaConf.set_struct(cfg, False)

    print(f"---------")
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"---------")

    settings = cfg.settings

    config_model = cfg.model
    model_name = config_model.name

    wandb_logger = WandbLogger(project='DL4CAST', name=f"{model_name}_synthetic",)
    wandb_logger.config = OmegaConf.to_container(cfg, resolve=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,    
        save_last=True,  
        #dirpath=f"/lustre/fsn1/projects/rech/ulm/uww31rp/checkpoints/patchfm",
        filename="patchfm-{epoch:02d}",
        save_on_train_epoch_end=True,
    )

    model = PatchFMLit(config=cfg.model) 

    trainset = artificial_dataset(seq_len=config_model.ws, target_len=config_model.patch_len, noise=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=settings.batch_size, shuffle=True,
                                                num_workers=settings.num_workers, pin_memory=settings.pin_memory)
    config_model["len_loader"] = len(trainloader)

    trainer = L.Trainer(
        max_epochs=config_model.epochs,
        enable_checkpointing=True,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        fast_dev_run=False,
        gradient_clip_val=config_model.max_norm if hasattr(config_model, "max_norm") else 0,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )
    trainer.fit(model=model, train_dataloaders=trainloader)

    wandb.finish()

if __name__ == '__main__':
    main()
