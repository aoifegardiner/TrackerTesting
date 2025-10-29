import logging
import re
from pathlib import Path
from . import MFT_files
logger = logging.getLogger(__name__)


class Config():
    def __init__(self):
        pass

    def __getattr__(self, name):
        # gets called on attempt to access not-existent attribute
        # C.foo.bar.baz does not fail if foo, foo.bar, or foo.bar.baz is not in config
        return Config()

    def __bool__(self):
        # the truth value should be false to accomodate for inexistent config values
        # e.g. C.foo.bar.baz == False if foo, foo.bar, or foo.bar.baz is not in config
        return False

    def merge(self, other, update_dicts=False):
        other_dict = other.__dict__
        other_keys = other_dict.keys()
        our_keys = self.__dict__.keys()
        for key in other_keys:
            if key in our_keys:
                if update_dicts and isinstance(key, dict) and isinstance(getattr(self, key), dict):
                    getattr(self, key).update(other_dict[key])
                else:
                    logger.debug(f"Rewriting key [{key}] in config. ({getattr(self, key)} -> {getattr(other, key)})")
                    setattr(self, key, other_dict[key])
            else:
                setattr(self, key, other_dict[key])

    def __repr__(self):
        return repr(self.__dict__)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


from pathlib import Path
import importlib.util

def load_config(path):
    path = Path(path)
    assert path.exists(), f"config {path} does not exist!"

    # Load the module from file
    spec = importlib.util.spec_from_file_location("tracker_config", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # ✅ Prioritize get_config first
    if hasattr(module, "get_config"):
        return module.get_config(path)
    elif hasattr(module, "WAFTConfig"):
        return module.WAFTConfig()
    else:
        raise AttributeError(f"Config file {path} has neither get_config() nor WAFTConfig()")
    


def config_file_from_template(path, out_path=None, **kwargs):
    assert Path(path).exists(), f"config {path} does not exist!"
    with open(path, 'r') as fin:
        contents = fin.read()

    for key, value in kwargs.items():
        pattern = f'___placeholder_{key}___'
        replacement = str(value)
        contents = re.sub(pattern, replacement, contents)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w') as fout:
            fout.write(contents)

    return contents
