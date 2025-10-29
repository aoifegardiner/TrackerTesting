# /Workspace/agardiner_STIR_submission/MFT_WAFT/MFT/MFT_files/configs/stir_waft_cfg.py
from MFT_WAFT.MFT.MFT import MFT
from pathlib import Path
from MFT_WAFT.MFT.config import Config

# ✅ Force import from your local WAFTConfig, not the package one
from MFT_WAFT.MFT.WAFT.config.WAFT_cfg import WAFTConfig

import numpy as np
import logging
logger = logging.getLogger(__name__)

def get_config(package_file, image_size_override=None):
    """
    Build STIR-style config but with WAFT instead of RAFT.
    """
    conf = Config()
    flow_conf = WAFTConfig()  # your local version with of_class()

    if image_size_override is not None:
        logger.info(f"Overriding WAFT image_size: {flow_conf.image_size} -> {image_size_override}")
        flow_conf.image_size = image_size_override

    conf.tracker_class = MFT
    conf.flow_config = flow_conf

    conf.deltas = [np.inf, 1, 2, 4, 8, 16, 32]
    conf.occlusion_threshold = 0.02
    conf.name = Path(__file__).stem

    return conf
