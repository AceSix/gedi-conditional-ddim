from dataset import create_dataset
from model.UNet import UNet
from utils.ema import EMA
from utils.engine import GaussianDiffusionTrainer
from utils.tools import train_one_epoch, validate_one_epoch, load_yaml
from utils.callbacks import ModelCheckpoint
import torch
from datetime import datetime


def train(config):
    consume = config["consume"]
    if consume:
        cp = torch.load(config["consume_path"])
        #config = cp["config"]
    print(config)

    device = torch.device(config["device"])
    train_loader = create_dataset(**config["TrainDataset"])
    val_loader = create_dataset(**config["ValDataset"])
    #test_loader = create_dataset(**config["TestDataset"])
    start_epoch = 1

    model = UNet(**config["Model"]).to(device)
    ema   = EMA(model, decay=0.9999)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=3e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    model_checkpoint = ModelCheckpoint(**config["Callback"])
    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1

    for epoch in range(start_epoch, config["epochs"] + 1):
        loss = train_one_epoch(trainer, train_loader, val_loader, optimizer, device, epoch, ema)

        if epoch % 2 == 0:
            # ------- validation every 3 epochs with EMA weights ----
            ema.store(model)
            val_loss, val_gap = validate_one_epoch(trainer, val_loader, device, epoch)
            ema.restore(model)
            print(f"[{datetime.now():%H:%M:%S}] Val loss: {val_loss:.5f}, Cond gap: {val_gap:.5f}")

        model_checkpoint.step(
            loss,
            model=model.state_dict(),
            config=config,
            optimizer=optimizer.state_dict(),
            start_epoch=epoch,
            model_checkpoint=model_checkpoint.state_dict()
        )
if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)