import sys
print(sys.path[:3])


import numpy as np
import jax
import jax.numpy as jnp
from tapnet.models import tapir_model
from tapnet.utils import model_utils


class TAPIRTrackerSurgT:
    """
    TAPIR-based monocular tracker for SurgT evaluation.
    Tracks a grid of query points inside the GT bbox and reconstructs a bounding box.
    """
    def __init__(self, im1, bbox1_gt, steps=5):
        # Load pretrained TAPIR checkpoint
        import os
        BASE = os.path.dirname(os.path.abspath(__file__))   # SurgT/src/
        ROOT = os.path.abspath(os.path.join(BASE, "..", "..", ".."))   # /Workspace/agardiner_STIR_submission
        print("BASE:", BASE)
        print("ROOT:", ROOT)
        ckpt_path = os.path.join(ROOT,"tapnet/checkpoints/causal_tapir_checkpoint.npy")
        print("Loading TAPIR checkpoint from:", ckpt_path)
        ckpt = np.load(ckpt_path, allow_pickle=True).item()
        #ckpt = np.load("tapnet/checkpoints/causal_tapir_checkpoint.npy", allow_pickle=True).item()
        self.params, self.state = ckpt["params"], ckpt["state"]
        self.tapir = tapir_model.ParameterizedTAPIR(
            params=self.params,
            state=self.state,
            tapir_kwargs=dict(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False),
        )

        # JIT-compile TAPIR inference
        self.online_init = jax.jit(self._init)
        self.online_predict = jax.jit(self._predict)

        # Initialize TAPIR query points (grid inside bbox)
        if bbox1_gt is not None:
            self.query_points = self._bbox_to_queries(bbox1_gt, steps=steps)
            self.query_features = self.online_init(
                frames=model_utils.preprocess_frames(im1[None, None]),
                points=self.query_points,
            )
            self.causal_state = self.tapir.construct_initial_causal_state(
                self.query_points.shape[1], len(self.query_features.resolutions) - 1
            )
        else:
            self.query_points = None
            self.query_features = None
            self.causal_state = None

    def _bbox_to_queries(self, bbox, steps=5):
        """Generate a regular grid of TAPIR query points inside bbox."""
        x, y, w, h = bbox
        xs = np.linspace(x, x + w - 1, steps)
        ys = np.linspace(y, y + h - 1, steps)
        xv, yv = np.meshgrid(xs, ys, indexing="xy")
        pts = np.stack([np.zeros_like(xv), yv, xv], axis=-1)
        return jnp.array(pts.reshape(-1, 3), dtype=jnp.float32)[None, :, :]

    def _init(self, frames, points):
        fg = self.tapir.get_feature_grids(frames, is_training=False)
        return self.tapir.get_query_features(
            frames, is_training=False, query_points=points, feature_grids=fg
        )

    def _predict(self, frames, features, causal_context):
        fg = self.tapir.get_feature_grids(frames, is_training=False)
        tr = self.tapir.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=fg,
            query_features=features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        causal_context = tr.pop("causal_context")
        return {k: v[-1] for k, v in tr.items()}, causal_context

    def _points_to_bbox_xywh(self, pts_np, H, W):
        """Compute tight bbox around points, clipped to image boundaries."""
        x_min = max(0, min(W - 1, float(pts_np[:, 0].min())))
        x_max = max(0, min(W - 1, float(pts_np[:, 0].max())))
        y_min = max(0, min(H - 1, float(pts_np[:, 1].min())))
        y_max = max(0, min(H - 1, float(pts_np[:, 1].max())))
        w = max(1.0, x_max - x_min)
        h = max(1.0, y_max - y_min)
        return [x_min, y_min, w, h]

    def tracker_update(self, im1):
        """Run TAPIR prediction for one new frame and return the predicted bbox."""
        if self.query_points is None:
            return None

        pred, self.causal_state = self.online_predict(
            frames=model_utils.preprocess_frames(im1[None, None]),
            features=self.query_features,
            causal_context=self.causal_state,
        )

        tracks = np.array(pred["tracks"][0, :, 0])     # (N, 2)
        occlusion = pred["occlusion"][0, :, 0]
        expected_dist = pred["expected_dist"][0, :, 0]

        # Combine occlusion + uncertainty
        visibles = model_utils.postprocess_occlusions(occlusion, expected_dist)
        visible_tracks = tracks[visibles]

        if len(visible_tracks) == 0:
            visible_tracks = tracks  # fallback if all occluded

        H, W = im1.shape[:2]
        return self._points_to_bbox_xywh(visible_tracks, H, W)
