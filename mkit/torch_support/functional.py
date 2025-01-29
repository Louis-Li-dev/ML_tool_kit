import torch
def layer_norm(x):
    with torch.no_grad():
        std = x.std(dim=1, keepdim=True)
        mean = x.mean(dim=1, keepdim=True)
        normed_x = (x - mean) / std
    return normed_x