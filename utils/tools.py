import math
import numpy as np
import torch
from torch.amp import autocast
import yaml
from pathlib2 import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple, Union
from torchvision.utils import make_grid
import os
import matplotlib.pyplot as plt

def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out

def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg


def train_one_epoch(trainer, loader, optimizer, scheduler, device, epoch, ema, grad_clip=1.0):
    
    trainer.train()
    train_loss = 0.0
    train_n = 0

    optimizer.zero_grad()
    
    # each batch
    with tqdm(loader) as data:
        for images, cond in data:
            
            x_0 = images.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)

            #loss = trainer(x_0, cond)
            with autocast('cuda', dtype=torch.bfloat16):
                loss = trainer(x_0, cond)
            loss.backward()

            # optional gradient-clipping 
            #torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            #ema.update(trainer.model)
            optimizer.zero_grad()

            # update stats (detach frees the graph)
            B = x_0.size(0)
            train_loss += loss.detach().cpu().numpy() * B
            train_n += B

            data.set_description(f"Epoch: {epoch}")
            data.set_postfix(ordered_dict={
                "train_loss": train_loss / train_n,
            })
            
    return train_loss / train_n




def validate_one_epoch(trainer, loader, device, epoch):
    
    # validation
    trainer.eval()

    val_loss = 0.0
    val_gap  = 0.0
    val_n = 0
    
    with torch.no_grad():
        with tqdm(loader) as data:
            for images, cond in data:
                x_0 = images.to(device, non_blocking=True)
                cond = cond.to(device, non_blocking=True)
                B = x_0.size(0)
    
                # -------- ordinary ε‑prediction loss -----------------------------
                with autocast('cuda', dtype=torch.bfloat16):
                    loss = trainer(x_0, cond, drop_prob=0.0)      # <‑‑ no dropout in eval
                    val_loss += loss.item() * B
    
                    t   = torch.randint(trainer.T, (B,), device=x_0.device)
                    eps = torch.randn_like(x_0, dtype=x_0.dtype)
    
                    # x_t from the forward‑diffusion equation
                    x_t = (extract(trainer.signal_rate, t, x_0.shape) * x_0 +
                            extract(trainer.noise_rate , t, x_0.shape) * eps)
    
                    eps_c  = trainer.model(x_t, t, cond)   # conditional prediction
                    null_cond = torch.ones_like(cond) * -1.0
                    eps_uc = trainer.model(x_t, t, null_cond)   # unconditional prediction
    
                    gap = (eps_c - eps_uc).pow(2).mean(dim=[1, 2]).sqrt().mean()  # scalar
                    val_gap += gap.item() * B
    
                val_n += B
            
    return val_loss/val_n, val_gap/val_n




def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None,
               format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image into a picture.

    Parameters:
        images: a tensor with shape (batch_size, channels, height, width).
        nrow: decide how many images per row. Default `8`.
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5
    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid


def save_sample_image(images: torch.Tensor, show: bool = True, path: Optional[str] = None,
                      format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    concat all image including intermediate process into a picture.

    Parameters:
        images: images including intermediate process,
            a tensor with shape (batch_size, sample, channels, height, width).
        show: whether to display the image after stitching. Default `True`.
        path: the path to save the image. if None (default), will not save image.
        format: image format. You can print the set of available formats by running `python3 -m PIL`.
        to_grayscale: convert PIL image to grayscale version of image. Default `False`.
        **kwargs: other arguments for `torchvision.utils.make_grid`.

    Returns:
        concat image, a tensor with shape (height, width, channels).
    """
    images = images * 0.5 + 0.5

    grid = []
    for i in range(images.shape[0]):
        # for each sample in batch, concat all intermediate process images in a row
        t = make_grid(images[i], nrow=images.shape[1], **kwargs)  # (channels, height, width)
        grid.append(t)
    # stack all merged images to a tensor
    grid = torch.stack(grid, dim=0)  # (batch_size, channels, height, width)
    grid = make_grid(grid, nrow=1, **kwargs)  # concat all batch images in a different row, (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid




def save_waveform_plot(waveforms: torch.Tensor, nrow: int = 8, show: bool = True, 
                       path: Optional[str] = None,
                       xlim: Optional[Tuple[float, float]] = None, 
                       ylim: Optional[Tuple[float, float]] = None,
                       figsize: Optional[Tuple[float, float]] = None,
                       title: Optional[str] = None):
    """
    Plot waveforms (line plots) in a grid with fixed x and y limits.
    
    Parameters:
        waveforms: a torch.Tensor of shape (batch_size, num_samples, num_channels, length).
                   For this function, num_samples is expected to be 1.
        nrow: number of waveforms per row (if multiple waveforms).
        show: whether to display the plot.
        path: file path to save the plot. If None, the plot is not saved.
        xlim: tuple (xmin, xmax) for x-axis limits. If None, defaults to (0, length-1).
        ylim: tuple (ymin, ymax) for y-axis limits. If None, no fixed y limits.
        figsize: figure size. If None, computed automatically.
    
    Returns:
        The matplotlib Figure object.
    """
    if waveforms.ndim != 4:
        raise ValueError("Expected waveforms tensor of shape (batch_size, num_samples, num_channels, length)")
    batch_size, num_samples, num_channels, length = waveforms.shape
    if num_samples != 1:
        raise ValueError(f"Expected num_samples==1 for save_waveform_plot, but got {num_samples}")
    
    # Remove the sample dimension so that each item is (num_channels, length)
    waveforms = waveforms.squeeze(1)  # now shape: (batch_size, num_channels, length)

    # Set default x-axis limits if not provided.
    if xlim is None:
        xlim = (0, length - 1)
    
    # Determine grid dimensions.
    ncols = nrow
    nrows = math.ceil(batch_size / ncols)
    if figsize is None:
        figsize = (ncols * 3, nrows * 2)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    if title is not None:
        fig.suptitle(title)
    
    x = np.arange(length)
    for i in range(batch_size):
        ax = axes[i]
        # waveform shape: (num_channels, length)
        waveform = waveforms[i].detach().cpu().numpy()
        # Plot each channel (overlayed)
        for ch in range(num_channels):
            ax.plot(x, waveform[ch], label=f"Ch {ch}")
        ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title(f"Waveform {i}")
        if num_channels > 1:
            ax.legend()
    
    # Hide extra axes if batch_size < nrows*ncols.
    for j in range(batch_size, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def save_sample_waveform_plot(waveforms: torch.Tensor, show: bool = True, 
                              path: Optional[str] = None,
                              xlim: Optional[Tuple[float, float]] = None, 
                              ylim: Optional[Tuple[float, float]] = None,
                              figsize: Optional[Tuple[float, float]] = None,
                              title: Optional[str] = None):
    """
    Plot sample waveforms (line plots) in a grid. Expects a tensor with shape 
    (batch_size, num_samples, num_channels, length). Each row in the grid corresponds 
    to a batch sample and each column corresponds to one of the intermediate process samples.
    
    Parameters:
        waveforms: a torch.Tensor of shape (batch_size, num_samples, num_channels, length)
        show: whether to display the plot.
        path: file path to save the plot. If None, the plot is not saved.
        xlim: tuple (xmin, xmax) for x-axis limits. If None, defaults to (0, length-1).
        ylim: tuple (ymin, ymax) for y-axis limits. If None, no fixed y limits.
        figsize: figure size. If None, computed automatically.
    
    Returns:
        The matplotlib Figure object.
    """
    
    if waveforms.ndim != 4:
        raise ValueError("Expected waveforms tensor of shape (batch_size, num_samples, num_channels, length)")
    batch_size, num_samples, num_channels, length = waveforms.shape

    if xlim is None:
        xlim = (0, length - 1)
    
    if figsize is None:
        figsize = (num_samples * 3, batch_size * 2)
    
    fig, axes = plt.subplots(batch_size, num_samples, figsize=figsize, squeeze=False)
    if title is not None:
        fig.suptitle(title)
        
    x = np.arange(length)
    for i in range(batch_size):
        for j in range(num_samples):
            ax = axes[i, j]
            # waveform shape: (num_channels, length)
            waveform = waveforms[i, j].detach().cpu().numpy()
            for ch in range(num_channels):
                ax.plot(x, waveform[ch], label=f"Ch {ch}")
            ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_title(f"Batch {i}, Sample {j}")
            if num_channels > 1:
                ax.legend()
    
    plt.tight_layout()
    
    if path is not None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
