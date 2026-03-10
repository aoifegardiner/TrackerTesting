import torch

sd = torch.load("/Workspace/agardiner_STIR_submission/checkpoints/50000_waft.pth", map_location="cpu")
sd = sd["state_dict"] if "state_dict" in sd else sd

def tensor_stats(t):
    return t.min().item(), t.max().item(), t.mean().item(), t.std().item()

for k in sd:
    if sd[k].ndim >= 2:   # skip scalars
        print(k, tensor_stats(sd[k]))
        break
