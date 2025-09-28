import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
import os

from model import PatchFMLit
from dataset import artificial_dataset, UTSDataset

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

    wandb_logger = WandbLogger(project='PatchFM', name=f"pretraining")
    wandb_logger.config = OmegaConf.to_container(cfg, resolve=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,    
        save_last=True,  
        dirpath=f"./ckpts/",
        filename="patchfm-{epoch:02d}",
        save_on_train_epoch_end=True,
    )

    model = PatchFMLit(config=cfg.model) 

    art_trainset = artificial_dataset(seq_len=config_model.ws, target_len=config_model.patch_len, noise=True)
    utsd_trainset = UTSDataset(input_len=config_model.ws, output_len=config_model.patch_len)
    trainset = torch.utils.data.ConcatDataset([art_trainset, utsd_trainset])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=settings.batch_size, shuffle=True,
                                                num_workers=settings.num_workers, pin_memory=settings.pin_memory)
    config_model["len_loader"] = len(trainloader)

    trainer = L.Trainer(
        max_epochs=config_model.epochs,
        enable_checkpointing=True,
        log_every_n_steps=10,
        accelerator="gpu",
        devices=int(os.environ['SLURM_GPUS_ON_NODE']),
        num_nodes=int(os.environ['SLURM_NNODES']),
        strategy="ddp",
        fast_dev_run=False,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )
    trainer.fit(model=model, train_dataloaders=trainloader)

    wandb.finish()

if __name__ == '__main__':
    main()
