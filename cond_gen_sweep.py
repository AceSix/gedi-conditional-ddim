import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torch.amp import autocast
from itertools import product
from tqdm import tqdm

from utils.engine import DDIMSampler
from utils.ema    import EMA
from model.UNet   import UNet


if __name__ == "__main__":
    # 1) Point this to your trained checkpoint
    checkpoint_path = "checkpoint/gedi_treecover_v6_e5.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Load checkpoint
    cp = torch.load(checkpoint_path, map_location=device)

    # 4) Instantiate model & load raw weights
    model = UNet(**cp["config"]["Model"]).to(device)
    model.load_state_dict(cp["model"])
    model.eval()

    # 5) Build EMA helper and inject saved shadow weights
    #ema = EMA(model, decay=0.9999)
    #ema.shadow = {k: v.to(device) for k, v in cp["ema_shadow"].items()}

    # 6) Swap in EMA weights
    #ema.store(model)    # backs up raw weights & copies EMA→model

    # 7) Define condition grid (0.0 → 100.0 over 21 points)
    cond_values = np.round(np.linspace(0.0, 100, 21), 2)

    # 8) Batch size & storage dict
    batch_size    = 100
    waveform_dict = {}

    # 9) Prepare common variables
    x_axis = np.arange(512)  # waveform length
    z_t    = torch.randn((batch_size, cp["config"]["Model"]["in_channels"], 512),
                         device=device)

    # 10) Sampling hyper-params
    steps_list      = [100, 250, 400]
    etas            = [0, 0.2, 0.5, 0.8, 1.0]
    guidance_scales = [0.5, 1.0, 1.5, 2.5]
    methods         = ["quadratic", "linear"]

    # Create a combined iterator over all configuration parameters.
    config_iter = list(product(steps_list, etas, guidance_scales, methods))
    for steps, eta, guidance_scale, method in tqdm(config_iter, desc="Configurations", leave=True):
        key = (steps, eta, guidance_scale, method)
        waveform_dict[key] = {}
        #print(f"Sampling for configuration {key}...")

        # Instantiate sampler for the current configuration
        sampler = DDIMSampler(
            model,
            guidance_scale=guidance_scale,
            **cp["config"]["Trainer"]
        ).to(device)

        # Inner loop: iterate through condition values with a progress bar.
        for cond in tqdm(cond_values, desc=f"Conditions for config {key}", leave=False, disable=True):
            cond_tensor = torch.full((batch_size, 1),
                                        cond,
                                        dtype=torch.float,
                                        device=device)
            with autocast('cuda', dtype=torch.bfloat16):
                x = sampler(
                    z_t, cond=cond_tensor,
                    only_return_x_0=True,
                    steps=steps, eta=eta,
                    method=method, progress_bar=False,
                )

            # Convert output to numpy & squeeze channel dimension
            waveforms = x.cpu().numpy().squeeze(1)  # shape: (batch_size, 512)
            waveform_dict[key][cond] = waveforms

    # Save the results to a pickle file.
    with open("waveform_results_sweeps.pkl", "wb") as f:
        pickle.dump(waveform_dict, f)

    print("Results have been saved to waveform_results.pkl")
