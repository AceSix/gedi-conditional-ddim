# utils/ema.py
import torch

class EMA:
    """
    Exponential Moving Average of model parameters.
    Call .update(model) after every optimizer step.
    Use .store(model) / .restore(model) to swap EMA weights in/out.
    """
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = {n: p.clone().detach()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n] = self.shadow[n] * self.decay + p.data * (1.0 - self.decay)

    # --- swap helpers -------------------------------------------------------
    def store(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = None
