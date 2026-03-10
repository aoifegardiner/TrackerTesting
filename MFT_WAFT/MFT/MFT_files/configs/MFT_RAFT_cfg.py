from MFT_WAFT.MFT.MFT import MFT
from pathlib import Path
from MFT_WAFT.MFT.config_RAFT import Config, load_config
import numpy as np

import logging
logger = logging.getLogger(__name__)


def get_config(package_file):
    conf = Config()

    conf.tracker_class = MFT
    print("tracker name", conf.tracker_class)
    conf.flow_config = load_config('MFT_files/configs/flow/RAFTou_kubric_huber_split_nonoccl.py')
    conf.deltas = [np.inf, 1, 2, 4, 8, 16, 32]
    #conf.deltas = [np.inf]
    conf.occlusion_threshold = 0.02

    conf.name = Path(__file__).stem
    print("conf name", conf.name)
    return conf