# MFT_files/configs/stir_waft_cfg.py  (example name)
from MFT.MFT import MFT
from pathlib import Path
from MFT.config import Config
from MFT.WAFT.config.WAFT_cfg import WAFTConfig  # your WAFTConfig

import numpy as np
import logging
    
#from MFT_WAFT.MFT.config import Config
#from MFT_WAFT.MFT.WAFT.config.WAFT_cfg import WAFTConfig  # your WAFTConfig

logger = logging.getLogger(__name__)

def get_config(package_file, image_size_override=None):
    """
    Build STIR-style config but with WAFT instead of RAFT.
    Optionally override image_size for demo/testing.
    """
    conf = Config()

    # Use WAFTConfig instead of RAFT-style config
    flow_conf = WAFTConfig()

    # Allow overriding image size for demo data
    if image_size_override is not None:
        logger.info(f"Overriding WAFT image_size: {flow_conf.image_size} -> {image_size_override}")
        flow_conf.image_size = image_size_override

    conf.tracker_class = flow_conf.tracker_class  # WAFTWrapper

    conf.flow_config = flow_conf  # now has algorithm, backbones, etc.
    conf.deltas = [np.inf, 1, 2, 4, 8, 16, 32]
    conf.occlusion_threshold = 0.02
    conf.name = Path(__file__).stem

    return conf
