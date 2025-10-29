from pathlib import Path
from MFT_WAFT.MFT.config_RAFT import Config
from MFT.raft import RAFTWrapper


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__.update(kwargs)


def get_config(packagefile):
    conf = Config()

    conf.of_class = RAFTWrapper
    conf_name = Path(__file__).stem

    raft_kwargs = {
        'occlusion_module': 'separate_with_uncertainty',
        'small': False,
        'mixed_precision': False,
    }
    conf.raft_params = AttrDict(**raft_kwargs)
    # original model location:
    conf.model = 'MFT_files/checkpoints/raft-things-sintel-kubric-splitted-occlusion-uncertainty-non-occluded-base-sintel.pth'
    conf.model = str(Path(packagefile, conf.model))

    conf.flow_iters = 12

    conf.flow_cache_dir = Path(f'flow_cache/{conf_name}/')
    conf.flow_cache_ext = '.flowouX16.pkl'
    conf.name = Path(__file__).stem

    return conf