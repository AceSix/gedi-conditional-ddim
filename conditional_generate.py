import math
import torch
import numpy as np
from argparse import ArgumentParser
from utils.ema import EMA
from utils.engine import DDPMSampler, DDIMSampler
from model.UNet import UNet
from utils.tools import save_sample_image, save_image, save_sample_waveform_plot, save_waveform_plot

def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    
    # generator parameters
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    
    # sampler parameters
    parser.add_argument("--result_only", default=False, action="store_true")
    parser.add_argument("--interval", type=int, default=50)
    
    # DDIM sampler parameters
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic", "karras"])
    
    # save image parameters
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)
    parser.add_argument("--to_grayscale", default=False, action="store_true")
    
    # gedi specific: generate waveforms
    parser.add_argument("--waveform", default=True, action="store_true")
    
    # New: Conditioning parameter -- a comma-separated list.
    # Example usage: "--cond 45.0,-120.0,350.0,0.7"
    parser.add_argument("--cond", type=str, default=None,
                        help="Comma separated list of conditioning values (e.g., longitude, latitude, elevation, tree cover)")
    
    args = parser.parse_args()
    return args

@torch.no_grad()
def generate(args):
    device = torch.device(args.device)
    
    cp = torch.load(args.checkpoint_path, map_location=device)
    # load trained model (ensure your model's forward accepts a condition argument)
    model = UNet(**cp["config"]["Model"]).to(device)
    model.load_state_dict(cp["model"])
    model.eval()
    
    if args.sampler == "ddim":
        sampler = DDIMSampler(model, **cp["config"]["Trainer"]).to(device)
    elif args.sampler == "ddpm":
        sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    ema = EMA(model, decay=0.9999)
    #ema.shadow = {k: v.to(device) for k, v in cp["ema_shadow"].items()}
    #ema.store(model)    # backs up raw weights & copies EMA→model
    
    # Parse the conditioning vector.
    # Expecting a comma-separated string converting to a list of floats.
    if args.cond is not None:
        try:
            cond_list = [float(v.strip()) for v in args.cond.split(",")]
            print(f"Condition list: {cond_list}")
        except Exception as e:
            raise ValueError("Could not parse condition. Make sure it is a comma separated list of numbers.") from e
    else:
        # If no condition is provided, use a default (e.g., zeros) with dimension defined in config (defaulting to 4)
        cond_dim = cp["config"]["Model"].get("cond_dim", 3)
        cond_list = [0.0] * cond_dim
    
    # Convert condition list to tensor and expand for the batch.
    cond_tensor = torch.tensor(cond_list, device=device, dtype=torch.float32)
    cond_tensor = cond_tensor.unsqueeze(0).repeat(args.batch_size, 1)  # [B, cond_dim]
    
    # Generate a title string to overlay on the plots.
    cond_title = "Condition: " + ", ".join(f"{v:.2f}" for v in cond_list)
    
    # Generate Gaussian noise.
    if not args.waveform:
        z_t = torch.randn(
            (args.batch_size, cp["config"]["Model"]["in_channels"],
             *cp["config"]["Dataset"]["image_size"]),
            device=device
        )
    else:
        z_t = torch.randn(
            (args.batch_size, cp["config"]["Model"]["in_channels"], 512),
            device=device
        )
        print(f"z_t shape: {z_t.shape}")
    
    extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
    x = sampler(z_t, cond=cond_tensor, only_return_x_0=args.result_only, interval=args.interval, **extra_param)
    x = torch.flip(x, dims=[-1])
    
    print(f"x shape: {x.shape}")
    
    # Save or display results. We'll now pass cond_title to the plotting functions.
    if not args.waveform:
        if args.result_only:
            save_image(x, nrow=args.nrow, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale, title=cond_title)
        else:
            save_sample_image(x, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale, title=cond_title)
    else:
        if args.result_only:
            save_waveform_plot(x, path=args.image_save_path, title=cond_title)
        else:
            save_sample_waveform_plot(x, path=args.image_save_path, title=cond_title)
    
if __name__ == "__main__":
    args = parse_option()
    generate(args)
