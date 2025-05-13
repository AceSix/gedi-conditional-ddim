import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from abc import ABC, abstractmethod


# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# use GN for norm layer, shape [B, C, L]
def norm_layer(channels):
    return nn.GroupNorm(num_groups=32, num_channels=channels, affine=True)


# Residual block adapted for 1D.
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        # Projection for time step embedding.
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        """
        x: [B, in_channels, L]
        t: [B, time_channels]
        """
        h = self.conv1(x)
        # Inject embedding (unsqueeze to add the temporal dimension).        
        e = self.emb(t)        
        h += e[:, :, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# Attention block adapted for 1D arrays.
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.norm = norm_layer(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, L = x.shape
        qkv = self.qkv(self.norm(x))
        head_dim = C // self.num_heads
        # Reshape to (B, num_heads, 3 * head_dim, L) and then split q, k, v.
        qkv = qkv.reshape(B, self.num_heads, 3 * head_dim, L)
        q, k, v = torch.chunk(qkv, 3, dim=2)  # Each: (B, num_heads, head_dim, L)
        #scale = 1. / math.sqrt(math.sqrt(head_dim))
        scale = 1. / math.sqrt(head_dim)
        
        # Compute attention.
        attn = torch.einsum("bncl,bncs->bnls", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bnls,bncs->bncl", attn, v)
        h = h.reshape(B, -1, L)
        h = self.proj(h)
        return h + x



# Upsample for 1D arrays.
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# Downsample for 1D arrays.
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        if use_conv:
            self.op = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


# UNet model adapted for 1D data.
class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            model_channels=128,
            out_channels=1,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 2, 2),
            conv_resample=True,
            num_heads=4,
            cond_dim=1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        self.time_embed_dim = model_channels * 4
        self.cond_embed_dim = model_channels // 2
        self.emb_dim = self.time_embed_dim + self.cond_embed_dim # concatenate time and cond embeddings

        # Time embedding network.        
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Conditioning embedding
        self.cond_embed = nn.Sequential(                   
            nn.Linear(cond_dim, self.cond_embed_dim),
            nn.SiLU(),
            nn.Linear(self.cond_embed_dim, self.cond_embed_dim),
            )


        # Downsample blocks.
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, self.emb_dim, dropout)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # Skip downsampling at the final stage.
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # Middle block.
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, self.emb_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, self.emb_dim, dropout)
        )

        # Upsample blocks.
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(ch + down_block_chans.pop(), model_channels * mult, self.emb_dim, dropout)
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        # Final output layer.
        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv1d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, cond):
        """
        x: [B, C, L] tensor of inputs.
        timesteps: a 1-D tensor of timesteps.
        cond: [B, cond_dim] tensor of conditioning inputs.
        The conditioning input is optional. If not provided, the model will
        not use any conditioning information.
        Returns: [B, out_channels, L] tensor.
        """
        hs = []
        # Compute time embedding.  shape: [B, time_embed_dim]
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        cond_emb = self.cond_embed(cond)                  # gradients update MLP
        emb = torch.cat([time_emb, cond_emb], dim=1)
        
        
        # Downsample stage.
        h = x
        
        #for module in self.down_blocks:
        for i, module in enumerate(self.down_blocks):
            h = module(h, emb)
            hs.append(h)
            
        # Middle stage.
        h = self.middle_block(h, emb)

        # Upsample stage.
        #for module in self.up_blocks:
        for i, module in enumerate(self.up_blocks):
            h = module(torch.cat([h, hs.pop()], dim=1), emb)

        return self.out(h)