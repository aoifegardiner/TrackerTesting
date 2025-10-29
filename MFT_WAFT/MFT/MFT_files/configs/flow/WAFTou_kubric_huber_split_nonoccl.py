from pathlib import Path
from MFT.config import Config
from MFT.waft import WAFTWrapper


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__.update(kwargs)


def get_config(packagefile):
    conf = Config()

    conf.of_class = WAFTWrapper
    conf_name = Path(__file__).stem

    raft_kwargs = {
        'occlusion_module': 'separate_with_uncertainty',
        'small': False,
        'mixed_precision': False,
    }
    conf.raft_params = AttrDict(**raft_kwargs)
    # original model location:
    conf.model = 'MFT/WAFT/ckpts/chairs-things.pth'
    #conf.model = str(Path(packagefile, conf.model))
    conf.model = str(Path(conf.model))

    conf.flow_iters = 12

    conf.flow_cache_dir = Path(f'flow_cache/{conf_name}/')
    conf.flow_cache_ext = '.flowouX16.pkl'
    conf.name = Path(__file__).stem

    return conf
