from pathlib import Path
from MFT_WAFT.MFT.config import Config
from MFT_WAFT.MFT.waft_ou import WAFTFlowOUForMFT
from MFT_WAFT.WAFT.model import build_waft_model   # <- whatever builds your WAFT core

import numpy as np
import logging
logger = logging.getLogger(__name__)

def get_config(package_file, image_size_override=None):
    conf = Config()

    # Build WAFT core model in the same way as train.py does
    waft_core = build_waft_model()  # or fetch_model(waft_args)

    # Wrap it for MFT
    flow_conf = Config()
    flow_conf.flow_model = WAFTFlowOUForMFT(waft_core)

    # Optional: image size override if WAFT needs it (for tiling/padding)
    if image_size_override is not None:
        flow_conf.image_size = image_size_override

    # tracker_class is the usual MFT tracker (FlowOU tracker)
    from MFT_WAFT.MFT.MFT import MFT
    conf.tracker_class = MFT
    conf.flow_config   = flow_conf

    conf.deltas = [np.inf, 1, 2, 4, 8, 16, 32]
    conf.occlusion_threshold = 0.02
    conf.name = Path(__file__).stem

    return conf
