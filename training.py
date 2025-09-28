import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
import wandb
from pytorch_lightning.loggers import WandbLogger
from pprint import pprint

from model import PatchFMLit
from dataset import get_dataset
from configs import TrainConfig, PatchFMConfig

def main():

    seed_everything(0, workers=True)

    train_cfg = TrainConfig()
    model_cfg = PatchFMConfig()

    assert train_cfg.seq_len % model_cfg.patch_len == 0, f"Sequence length ({train_cfg.seq_len}) must be divisible by patch length ({model_cfg.patch_len})."

    print("---------")
    print("Model Configuration:")
    pprint(model_cfg.__dict__, sort_dicts=False)

    print("Training Configuration:")
    pprint(train_cfg.__dict__, sort_dicts=False)
    print("---------")

    wandb_logger = WandbLogger(project='PatchFM', name=f"pretraining")

    wandb_logger.model_config = model_cfg
    wandb_logger.train_config = train_cfg

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,    
        save_last=True,  
        dirpath=train_cfg.checkpoint_path,
        filename="patchfm-{epoch:02d}",
        save_on_train_epoch_end=True,
    )

    model = PatchFMLit(train_config=train_cfg, model_config=model_cfg) 

    trainset = get_dataset(seq_len=train_cfg.seq_len, target_len=model_cfg.patch_len)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_cfg.batch_size, shuffle=True,
                                                num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory)
    train_cfg.len_loader = len(trainloader)

    trainer = L.Trainer(
        max_epochs=train_cfg.epochs,
        enable_checkpointing=True,
        log_every_n_steps=10,
        accelerator="gpu",
        devices=train_cfg.gpus,
        num_nodes=train_cfg.num_nodes,
        strategy=train_cfg.strategy,
        fast_dev_run=False,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )
    trainer.fit(model=model, train_dataloaders=trainloader)

    wandb.finish()

if __name__ == '__main__':
    main()
