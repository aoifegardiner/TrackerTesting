# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import sys
sys.path.insert(0, "/Workspace/agardiner_STIR_submission/MFT_WAFT")  # directory that contains the MFT/ package

import einops
import numpy as np
import torch
from types import SimpleNamespace
import logging
#from MFT_WAFT.MFT.waft  import FlowOUTrackingResult as FlowOUTrackingResult
from MFT_WAFT.MFT.results import FlowOUTrackingResult


from MFT_WAFT.MFT.utils.timing import general_time_measurer

logger = logging.getLogger(__name__)

import os
import cv2
import numpy as np
import torch

def flow_to_color(flow, max_flow=None):
    """Convert optical flow (2,H,W) to color (H,W,3) for visualization."""
    flow = flow.detach().cpu().numpy()
    flow = np.transpose(flow, (1, 2, 0))  # (H, W, 2)
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    if max_flow is None:
        max_flow = np.percentile(mag, 99)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang / 2  # Hue: direction
    hsv[..., 1] = 255      # Saturation: full
    hsv[..., 2] = np.clip((mag / (max_flow + 1e-5)) * 255, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

import torch
from pathlib import Path

class DiskFlowCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, left_id, right_id):
        return self.root / f"{left_id:06d}_{right_id:06d}.pt"

    def write(self, left_id, right_id, flow, occ, sigma):
        path = self._path(left_id, right_id)

        torch.save({
            "flow": flow.detach().cpu(),
            "occlusion": occ.detach().cpu(),
            "sigma": sigma.detach().cpu(),
        }, path)

    def read(self, left_id, right_id, device=None):
        path = self._path(left_id, right_id)
        if not path.exists():
            raise FileNotFoundError

        data = torch.load(path, map_location="cpu")

        flow = data["flow"].float()
        occ = data["occlusion"].float()
        sig = data["sigma"].float()

        if device is not None:
            flow = flow.to(device)
            occ  = occ.to(device)
            sig  = sig.to(device)

        return flow, occ, sig


class MFT():
    def __init__(self, config):
        
        """Create MFT tracker
        args:
          config: a MFT.config.Config, for example from configs/MFT_cfg.py"""
        self.C = config   # must be named self.C, will be monkeypatched!
        self.flower = config.flow_config.of_class(config.flow_config)  # init the OF
        import inspect
        print("flower type:", type(self.flower))
        print("flower file:", inspect.getfile(type(self.flower)))
        self.device = 'cuda'

    def init(self, img, start_frame_i=0, time_direction=1, flow_cache=None, **kwargs):
        """Initialize MFT on first frame

        args:
          img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          start_frame_i: [optional] init frame number (used for caching)
          time_direction: [optional] forward = +1, or backward = -1 (used for caching)
          flow_cache: [optional] MFT.utils.io.FlowCache (for caching OF on GPU, RAM, or SSD)
          kwargs: [unused] - for compatibility with other trackers

        returns:
          meta: initial frame result container, with initial (zero-motion) MFT.results.FlowOUTrackingResult in meta.result 
        """
        print("Starting MFT track")
        self.img_H, self.img_W = img.shape[:2]
        self.start_frame_i = start_frame_i
        self.current_frame_i = self.start_frame_i
        assert time_direction in [+1, -1]
        self.time_direction = time_direction
        self.flow_cache = flow_cache or DiskFlowCache("./flow_cache_MFT_Stir")
        #self.flow_cache = flow_cache

        self.memory = {
            self.start_frame_i: {
                'img': img,
                'result': FlowOUTrackingResult.identity((self.img_H, self.img_W), device=self.device)
            }
        }

        self.template_img = img.copy()

        meta = SimpleNamespace()
        meta.result = self.memory[self.start_frame_i]['result'].clone().cpu()
        return meta


    def track(self, input_img, debug=False, **kwargs):
        """Track one frame

        args:
          input_img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          debug: [optional] enable debug visualizations
          kwargs: [unused] - for compatibility with other trackers

        returns:
          meta: current frame result container, with MFT.results.FlowOUTrackingResult in meta.result
                The meta.result represents the accumulated flow field from the init frame, to the current frame
        """
        meta = SimpleNamespace()
        self.current_frame_i += self.time_direction
        #print("Startig MFT track")
        # OF(init, t) candidates using different deltas
        delta_results = {}
        already_used_left_ids = []
        chain_timer = general_time_measurer('chain', cuda_sync=True, start_now=False, active=self.C.timers_enabled)
        for delta in self.C.deltas:
        #for delta in [1]:
            # candidates are chained from previous result (init -> t-delta) and flow (t-delta -> t)
            # when tracking backward, the chain consists of previous result (init -> t+delta) and flow(t+delta -> t)
            left_id = self.current_frame_i - delta * self.time_direction
            right_id = self.current_frame_i

            # we must ensure that left_id is not behind the init frame
            if self.is_before_start(left_id):
                if np.isinf(delta):
                    left_id = self.start_frame_i
                else:
                    continue
            left_id = int(left_id)

            # because of this, different deltas can result in the same left_id, right_id combination
            # let's not recompute the same candidate multiple times
            if left_id in already_used_left_ids:
                continue

            left_img = self.memory[left_id]['img']
            right_img = input_img

            template_to_left = self.memory[left_id]['result']

            flow_init = None
            use_cache = np.isfinite(delta) or self.C.cache_delta_infinity
            cache = DiskFlowCache("./flow_cache_MFT_Stir")
            left_to_right = get_flowou_with_cache(self.flower, left_img, right_img, flow_init,
                                                  cache=cache, left_id=left_id, right_id=right_id,
                                                  read_cache=True, write_cache=True)

            chain_timer.start()
            delta_results[delta] = chain_results(template_to_left, left_to_right)

            def mean1(x): 
                return x.mean().item()
            
            chained = chain_results(template_to_left, left_to_right)
            delta_results[delta] = chained
            
            print(
                f"delta={delta} left_id={left_id} | "
                f"sigma(init->left)={mean1(template_to_left.sigma):.3f}  "
                f"sigma(left->right)={mean1(left_to_right.sigma):.3f}  "
                f"sigma(chained)={mean1(chained.sigma):.3f}  "
                f"occ(edge)={mean1(left_to_right.occlusion):.3f}"
            )


            already_used_left_ids.append(left_id)
            chain_timer.stop()

        chain_timer.report('mean')
        chain_timer.report('sum')

        selection_timer = general_time_measurer('selection', cuda_sync=True, start_now=True,
                                                active=self.C.timers_enabled)
        used_deltas = sorted(list(delta_results.keys()), key=lambda delta: 0 if np.isinf(delta) else delta)
        all_results = [delta_results[delta] for delta in used_deltas]
        all_flows = torch.stack([result.flow for result in all_results], dim=0)  # (N_delta, xy, H, W)
        all_sigmas = torch.stack([result.sigma for result in all_results], dim=0)  # (N_delta, 1, H, W)
        all_occlusions = torch.stack([result.occlusion for result in all_results], dim=0)  # (N_delta, 1, H, W)


        #scores = -all_sigmas
        #scores[all_occlusions > self.C.occlusion_threshold] = -float('inf')

#        sigma_norm = all_sigmas / (all_sigmas.mean(dim=0, keepdim=True) + 1e-6)
#        scores = -sigma_norm
#        scores[all_occlusions > self.C.occlusion_threshold] = -float('inf')


        scores = -all_sigmas
        #scores[all_occlusions > self.C.occlusion_threshold] = -float('inf')
        best = scores.max(dim=0, keepdim=True)
        selected_delta_i = best.indices  # (1, 1, H, W)
        best_flow = all_flows.gather(dim=0,
                                     index=einops.repeat(selected_delta_i,
                                                         'N_delta 1 H W -> N_delta xy H W',
                                                         xy=2, H=self.img_H, W=self.img_W))
        best_occlusions = all_occlusions.gather(dim=0, index=selected_delta_i)
        best_sigmas = all_sigmas.gather(dim=0, index=selected_delta_i)
        selected_flow, selected_occlusion, selected_sigmas = best_flow, best_occlusions, best_sigmas

        selected_flow = einops.rearrange(selected_flow, '1 xy H W -> xy H W', xy=2, H=self.img_H, W=self.img_W)
        selected_occlusion = einops.rearrange(selected_occlusion, '1 1 H W -> 1 H W', H=self.img_H, W=self.img_W)
        selected_sigmas = einops.rearrange(selected_sigmas, '1 1 H W -> 1 H W', H=self.img_H, W=self.img_W)

        result = FlowOUTrackingResult(selected_flow, selected_occlusion, selected_sigmas)
        # mark flows pointing outside of the current image as occluded
        invalid_mask = einops.rearrange(result.invalid_mask(), 'H W -> 1 H W')
        result.occlusion[invalid_mask] = 1
        selection_timer.report()

        meta.selected_delta_i = selected_delta_i.detach().cpu()   # int64
        meta.used_deltas = used_deltas 

        if getattr(self.C, "store_delta_value_map", False):
            # build lookup as float tensor
            # represent inf as a large sentinel or np.inf (torch supports inf in float)
            lookup = torch.tensor(
                [float(d) if np.isfinite(d) else float("inf") for d in used_deltas],
                device=selected_delta_i.device,
                dtype=torch.float32
            )  # (D,)
            delta_value_map = lookup[selected_delta_i.squeeze(0).squeeze(0)]  # (H,W)
            meta.delta_value_map = delta_value_map.detach().cpu()             # (H,W)

    
        out_result = result.clone()
            
        meta.result = out_result
        meta.result.cpu()

        self.memory[self.current_frame_i] = {'img': input_img,
                                             'result': result}

        self.cleanup_memory()
        return meta

    # @profile
    def cleanup_memory(self):
        # max delta, ignoring the inf special case
        try:
            max_delta = np.amax(np.array(self.C.deltas)[np.isfinite(self.C.deltas)])
        except ValueError:  # only direct flow
            max_delta = 0
        has_direct_flow = np.any(np.isinf(self.C.deltas))
        memory_frames = list(self.memory.keys())
        for mem_frame_i in memory_frames:
            if mem_frame_i == self.start_frame_i and has_direct_flow:
                continue

            if self.time_direction > 0 and mem_frame_i + max_delta > self.current_frame_i:
                # time direction     ------------>
                # mem_frame_i ........ current_frame_i ........ (mem_frame_i + max_delta)
                # ... will be needed later
                continue

            if self.time_direction < 0 and mem_frame_i - max_delta < self.current_frame_i:
                # time direction     <------------
                # (mem_frame_i - max_delta) ........ current_frame_i .......... mem_frame_i
                # ... will be needed later
                continue

            del self.memory[mem_frame_i]

    def is_before_start(self, frame_i):
        return ((self.time_direction > 0 and frame_i < self.start_frame_i) or  # forward
                (self.time_direction < 0 and frame_i > self.start_frame_i))    # backward


# @profile
def get_flowou_with_cache(flower, left_img, right_img, flow_init=None,
                          cache=None, left_id=None, right_id=None,
                          read_cache=False, write_cache=False, 
                          compose_on_miss=True):
    """Compute flow from left_img to right_img. Possibly with caching.

    Returns:
        flowou: FlowOUTrackingResult(flow, occlusion, sigma)
    """
    import os, cv2, numpy as np, torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_device_f32(flow, occ, sig):
        return flow.float().to(device), occ.float().to(device), sig.float().to(device)


#    must_compute = not read_cache
#    flow_left_to_right = occlusions = sigmas = None

    # Try reading from cache if allowed
    if read_cache and flow_init is None and cache is not None:
        try:
            flow_left_to_right, occlusions, sigmas = cache.read(left_id, right_id, device=device)
            flow_left_to_right, occlusions, sigmas = _to_device_f32(flow_left_to_right, occlusions, sigmas)
            print(f"[CACHE HIT] {left_id}->{right_id}")
            return FlowOUTrackingResult(flow_left_to_right, occlusions, sigmas)
        except Exception:
            print(f"[CACHE MISS] {left_id}->{right_id}")

    
#    if compose_on_miss and read_cache and flow_init is None and cache is not None:
#        if left_id is not None and right_id is not None and right_id > left_id:
#            try:
#                # first edge
#                f, o, s = cache.read(left_id, left_id + 1, device=device)
#                f, o, s = _to_device_f32(f, o, s)
#                composed = FlowOUTrackingResult(f, o, s)
#
#                # chain remaining edges
#                for k in range(left_id + 1, right_id):
#                    f, o, s = cache.read(k, k + 1, device=device)
#                    f, o, s = _to_device_f32(f, o, s)
#                    edge = FlowOUTrackingResult(f, o, s)
#                    composed = chain_results(composed, edge)
#
#                print(f"[CACHE COMPOSED] {left_id}->{right_id} from edges")
#                return composed
#            except Exception:
#                # fall through to compute
#                pass

    flow, extra = flower.compute_flow(left_img, right_img, mode="flow")
    occ = extra["occlusion"]
    sig = extra["sigma"]

    # IMPORTANT: make sure these are float32 on CUDA for chaining
    flow = flow.float().to(device)
    occ  = occ.float().to(device)
    sig  = sig.float().to(device)

    # ---- 4) Write only consecutive to disk (half precision to save space)
    if cache is not None and write_cache and flow_init is None:
        if right_id == left_id + 1:
            cache.write(left_id, right_id, flow.half().cpu(), occ.half().cpu(), sig.half().cpu())

    return FlowOUTrackingResult(flow, occ, sig)


def chain_results(left_result, right_result):
    flow = left_result.chain(right_result.flow)
    occlusions = torch.maximum(left_result.occlusion,
                               left_result.warp_backward(right_result.occlusion))
    sigmas = torch.sqrt(torch.square(left_result.sigma) +
                        torch.square(left_result.warp_backward(right_result.sigma)))
    return FlowOUTrackingResult(flow, occlusions, sigmas)

