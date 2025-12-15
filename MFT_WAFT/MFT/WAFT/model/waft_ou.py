import torch
import torch.nn as nn

class WAFT_OU(nn.Module):
    def __init__(self, flow_model, args):
        super().__init__()
        self.flow_model = flow_model
        self.args = args

        # 4 channels = info head channels
        # Output 2 channels = logits for {non-occ, occ}
        self.occlusion_head = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)   # <<< FIX HERE
        )

        # Uncertainty stays 1-channel
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, image1, image2, iters=None, flow_gt=None, **kwargs):
        kwargs.pop("test_mode", None)
        out = self.flow_model(image1, image2, iters=iters, flow_gt=flow_gt)

        # (B,4,H,W)
        info = out["info"][-1]

        occ_logits = self.occlusion_head(info)
        unc_pred   = self.uncertainty_head(info)

        # Provide at least 2 predictions to satisfy RAFT-style iteration
        out["occlusion"]   = [occ_logits, occ_logits.clone()]
        out["uncertainty"] = [unc_pred,   unc_pred.clone()]

        return out
