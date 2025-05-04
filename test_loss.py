import argparse
import torch
from tqdm import tqdm
from utils.tools import load_yaml
from dataset import create_dataset
from model.UNet import UNet
from utils.engine import GaussianDiffusionTrainer

def parse_args():
    p = argparse.ArgumentParser(description="Compute DDIM/DDPM loss + waveform correlation on a split")
    p.add_argument("-cp", "--checkpoint_path", type=str, required=True,
                   help="Path to your .pth checkpoint")
    p.add_argument("-cfg", "--config", type=str, default="config.yml",
                   help="YAML with TrainDataset/ValDataset/TestDataset sections")
    p.add_argument("-s", "--split", type=str, choices=["train","val","test"],
                   default="test", help="Which split to evaluate")
    p.add_argument("--device", type=str, default="cuda",
                   help="torch device")
    return p.parse_args()

def pearson_corr(a: torch.Tensor, b: torch.Tensor, dim: int = 1) -> torch.Tensor:
    # a, b shape: [batch, ...], collapse along 'dim'
    a_mean = a.mean(dim=dim, keepdim=True)
    b_mean = b.mean(dim=dim, keepdim=True)
    a_cent = a - a_mean
    b_cent = b - b_mean
    num = (a_cent * b_cent).sum(dim=dim)
    denom = torch.sqrt((a_cent**2).sum(dim=dim) * (b_cent**2).sum(dim=dim))
    return num / (denom + 1e-8)

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load config
    cfg = load_yaml(args.config)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    config = ckpt.get("config", cfg)

    # Model + trainer
    model = UNet(**config["Model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    # DataLoader
    ds_key = f"{args.split.capitalize()}Dataset"
    if ds_key not in config:
        raise KeyError(f"{ds_key} not found in config.yml")
    loader = create_dataset(**config[ds_key])

    total_loss = 0.0
    total_corr = 0.0
    total_n = 0

    with torch.no_grad():
        for x, cond in tqdm(loader, desc=f"Evaluating {args.split}"):
            x = x.to(device, non_blocking=True)        # shape [B, C, T] or [B, T]
            cond = cond.to(device, non_blocking=True)

            # 1) compute training loss
            loss = trainer(x, cond)
            total_loss += loss.item() * x.size(0)

            # 2) sample a waveform from the model given the condition
            #    assumes trainer.sample returns same shape as x
            pred = trainer.sample(cond, shape=x.shape).to(device)

            # 3) compute Pearson correlation on flattened time dimension
            #    collapse channel+time dims into one
            pred_flat = pred.flatten(start_dim=1)
            x_flat = x.flatten(start_dim=1)
            corr = pearson_corr(pred_flat, x_flat, dim=1)  # [B]
            total_corr += corr.sum().item()

            total_n += x.size(0)

    avg_loss = total_loss / total_n
    avg_corr = total_corr / total_n

    print(f"{args.split.capitalize()} loss: {avg_loss:.6f}")
    print(f"{args.split.capitalize()} waveform corr: {avg_corr:.6f}")

if __name__ == "__main__":
    main()
