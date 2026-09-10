"""
Run MapAnything on a 6 GB GPU (RTX 2060, sm_75 / Turing).

Why this file exists
--------------------
MapAnything's encoder is DINOv2 ViT-g/14. The checkpoint is 4.9 GB in fp32,
which leaves ~0.4 GB on a 6.4 GB card -- inference cannot run at all. The
weights must be held in 16-bit.

fp16 does NOT work: the ViT-g activations exceed fp16's exponent range and
every output comes back NaN (measured: 456876/456876 ray values NaN).
bf16 has fp32's range and works. Turing has no native bf16 tensor cores, so
it runs via emulation -- slower, but correct, and correctness is the gate here.

Three upstream assumptions break when weights are not fp32:
  1. Some submodules build fp32 tensors internally and feed them to Linear
     layers outside any autocast region (camera quat/trans encoder,
     model.py:1140). A global forward pre-hook bridges the dtype.
  2. geometry.py builds its pose matrix with torch.eye(4, device=device) and
     no dtype=, so it stays fp32 while points are 16-bit. Geometry and
     intrinsics recovery are forced to fp32 -- correct anyway, since 16-bit
     metric coordinates lose real precision.
  3. postprocess calls .cpu().numpy(), and numpy has no bfloat16. Outputs are
     upcast to fp32 before postprocessing.

Do not pass use_amp=True here. The weights are already 16-bit; autocast on top
adds nothing and reintroduces dtype conflicts.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import mapanything.models.mapanything.model as _model
import mapanything.utils.geometry as _geo
import mapanything.utils.inference as _inf
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# Measured on a 6.4 GB RTX 2060 with ~0.4 GB held by the Windows desktop.
# The binding constraint is total pixels per forward pass, not view count:
# 4x(294x518)=609k ok / 6x(252x448)=677k ok / 10x(182x336)=611k ok
# 16x(154x280)=690k ok / 24x(126x224)=677k ok / 12x(182x336)=733k OOM
PIXEL_BUDGET = 700_000

_patched = False


def _apply_patches(dtype):
    global _patched
    if _patched:
        return
    _patched = True

    def pre_hook(module, args):
        p = next(module.parameters(recurse=False), None)
        if p is None or p.dtype != dtype:
            return None
        changed, out = False, []
        for a in args:
            if torch.is_tensor(a) and a.dtype == torch.float32:
                a = a.to(dtype)
                changed = True
            out.append(a)
        return tuple(out) if changed else None

    torch.nn.modules.module.register_module_forward_pre_hook(pre_hook)

    _pm = _geo.convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap

    def fp32_pointmap(rd, dar, pt, pq):
        return _pm(rd.float(), dar.float(), pt.float(), pq.float()).to(rd.dtype)

    _geo.convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap = fp32_pointmap
    _model.convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap = fp32_pointmap

    _ri = _geo.recover_pinhole_intrinsics_from_ray_directions

    def fp32_intrinsics(rays, *a, **k):
        return _ri(rays.float(), *a, **k)

    _geo.recover_pinhole_intrinsics_from_ray_directions = fp32_intrinsics
    _inf.recover_pinhole_intrinsics_from_ray_directions = fp32_intrinsics

    _pp = _inf.postprocess_model_outputs_for_inference

    def up(x):
        if torch.is_tensor(x) and x.dtype in (torch.bfloat16, torch.float16):
            return x.float()
        if isinstance(x, dict):
            return {k: up(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(up(v) for v in x)
        return x

    def wrapped(raw_outputs, input_views, *a, **k):
        return _pp(up(raw_outputs), up(input_views), *a, **k)

    _inf.postprocess_model_outputs_for_inference = wrapped
    _model.postprocess_model_outputs_for_inference = wrapped


def load_model(repo="facebook/map-anything-apache", dtype=torch.bfloat16):
    """Load MapAnything with 16-bit weights.

    The cast happens on the CPU before .to("cuda") on purpose: casting after
    the move leaves the discarded fp32 blocks in the caching allocator, and
    the card ends up with 2.46 GB of live weights but only 0.39 GB free.
    """
    _apply_patches(dtype)
    model = MapAnything.from_pretrained(repo).eval().to(dtype).to("cuda")
    torch.cuda.empty_cache()
    return model


def max_views(longest_side, aspect=16 / 9, budget=PIXEL_BUDGET):
    """How many views fit in one forward pass at a given resolution."""
    h = int(longest_side / aspect)
    return max(1, budget // (h * longest_side))


def infer(model, image_paths, longest_side=None, **kw):
    """Run inference on one chunk. Raises if the chunk exceeds the budget."""
    if longest_side:
        views = load_images(image_paths, resize_mode="longest_side", size=longest_side)
    else:
        views = load_images(image_paths)

    h, w = views[0]["img"].shape[-2:]
    total = h * w * len(views)
    if total > PIXEL_BUDGET:
        raise ValueError(
            f"{len(views)} views at {h}x{w} = {total:,} px exceeds the "
            f"{PIXEL_BUDGET:,} px budget for this GPU. Use fewer views or a "
            f"smaller longest_side."
        )

    opts = dict(memory_efficient_inference=True, minibatch_size=1, use_amp=False)
    opts.update(kw)
    with torch.no_grad():
        return model.infer(views, **opts)
