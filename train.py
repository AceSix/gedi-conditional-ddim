import sched
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
    ##ema   = EMA(model, decay=0.9999)
    ema = None
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config["lr"], 
                                  betas=(0.9, 0.999), 
                                  weight_decay=3e-4)

    steps_per_epoch = len(train_loader)
    total_steps = config["epochs"] * steps_per_epoch
    warmup_steps = 0.1 * total_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler = None
    
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    model_checkpoint = ModelCheckpoint(**config["Callback"])
    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1

    for epoch in range(start_epoch, config["epochs"] + 1):
        loss = train_one_epoch(trainer, train_loader, optimizer, scheduler, device, epoch, ema)
        
        if epoch % 1 == 0:
            # ------- validation every 1 epochs with EMA weights ----
            #ema.store(model)
            val_loss, val_gap = validate_one_epoch(trainer, val_loader, device, epoch)
            #ema.restore(model)
            print(f"[{datetime.now():%H:%M:%S}] Val loss: {val_loss:.5f}, Cond gap: {val_gap:.5f}")

        model_checkpoint.step(
            loss,
            model=model.state_dict(),
            #ema_shadow=ema.shadow,
            config=config,
            optimizer=optimizer.state_dict(),
            start_epoch=epoch,
            model_checkpoint=model_checkpoint.state_dict()
        )
        
if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)