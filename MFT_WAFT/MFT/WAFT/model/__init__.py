#from MFT_WAFT.MFT.WAFT.model.vitwarp_v8 import ViTWarpV8
#from MFT_WAFT.MFT.WAFT.model.waft_ou import WAFT_OU  # new file
#
#def fetch_model(args):
#    algo = getattr(args, "algorithm", "waft").lower()
#    if algo in ("vitwarp", "vitwarp_v8"):
#        return ViTWarpV8(args)
#    elif algo in ("waft_ou", "waft_uncertainty", "waft_occlusion"):
#        return WAFT_OU(args)
#    else:
#        raise ValueError(f"Unknown algorithm: {algo}")



from .vitwarp_v8 import ViTWarpV8
from .waft_ou import WAFT_OU_FromFlow

def fetch_model(args):
    base_model = ViTWarpV8(args)
    wrapped = WAFT_OU_FromFlow(base_model, args)
    return wrapped
