from pathlib import Path
import inspect, sys
print("[DEBUG] Using WAFTWrapper module from:", sys.modules.get('MFT.WAFT.waft_wrapper'))


class WAFTConfig:
    def __init__(self):
        HERE = Path(__file__).resolve().parent  
        PROJECT_ROOT = Path("/Workspace/agardiner_STIR_submission")
        
        # path to WAFT JSON config
        self.waft_json = PROJECT_ROOT / "MFT_WAFT" / "MFT" / "WAFT" / "config" / "waft_stir_test.json"

        # checkpoint path
        self.ckpt = PROJECT_ROOT / "MFT_WAFT" / "MFT" / "WAFT" / "ckpts" / "chairs-things.pth"

        # dataset for evaluation
        self.dataset = "stir"

        # image size
        self.image_size = (1024, 1280)

        # WAFT scale factor
        self.scale = 0

        # algorithm name required by fetch_model
        self.algorithm = "vitwarp"

        # backbones required by ViTWarpV8
        self.dav2_backbone = "vits"
        self.network_backbone = "vits"

        self.iters = 12 

        # choose WAFT wrapper
        from MFT.waft import WAFTWrapper
        self.tracker_class = WAFTWrapper
    
    def of_class(self, config):
        """Factory method for compatibility with MFT"""
        # choose WAFT wrapper
        from MFT.waft import WAFTWrapper
        self.tracker_class = WAFTWrapper
        return self.tracker_class(config)
