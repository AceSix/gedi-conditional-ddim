import argparse
import torch
from tqdm import tqdm
from utils.tools import load_yaml
from dataset import create_dataset
from model.UNet import UNet
from utils.engine import GaussianDiffusionTrainer

def parse_args():
    p = argparse.ArgumentParser(description="Compute DDIM/ DDPM loss on a dataset split")
    p.add_argument("-cp", "--checkpoint_path", type=str, required=True,
                   help="Path to your .pth checkpoint")
    p.add_argument("-cfg", "--config", type=str, default="config.yml",
                   help="YAML with TrainDataset/ValDataset/TestDataset sections")
    p.add_argument("-s", "--split", type=str, choices=["train","val","test"],
                   default="test", help="Which split to evaluate")
    p.add_argument("--device", type=str, default="cuda",
                   help="torch device")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load config (allow override from checkpoint if present)
    cfg = load_yaml(args.config)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    config = ckpt.get("config", cfg)

    # Model + trainer setup
    model = UNet(**config["Model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    # DataLoader for the chosen split
    ds_key = f"{args.split.capitalize()}Dataset"  # e.g. "ValDataset"
    if ds_key not in config:
        raise KeyError(f"{ds_key} not found in config.yml")
    loader = create_dataset(**config[ds_key])

    # Loop and accumulate loss
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, cond in tqdm(loader, desc=f"Evaluating {args.split}"):
            x = x.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            loss = trainer(x, cond)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)

    print(f"{args.split.capitalize()} loss: {total_loss / n:.6f}")

if __name__ == "__main__":
    main()
