#!/usr/bin/env python
"""
3D Gaussian Splatting from a MapAnything scene, sized for a 6 GB GPU.

    python pipeline/train_gsplat.py scene_train/ -o splat/ --iters 7000

Consumes the reconstruction pipeline's own output -- poses from scene.json,
initial geometry from points.ply -- so there is no COLMAP round trip.

Cameras stay in MapAnything's native OpenCV axes (+X right, +Y down,
+Z forward), which is exactly what gsplat's viewmats expect. Do NOT point this
at a scene built with --y-up: that rotation exists for viewers, and here it
would flip every camera upside down.

Gaussian count is capped (MCMC strategy) rather than grown freely, because on
6 GB the count is the thing that decides whether training fits at all.
"""

import argparse
import json
import math
import os
import sys

import time

import numpy as np
import torch
import torch.nn.functional as F

from gsplat import export_splats, rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

SH_C0 = 0.28209479177387814
BASE_W = 518.0  # width MapAnything's predicted intrinsics refer to
BASE_H = 294.0  # matching height (16:9 mapping)


def require_cuda_toolkit():
    """gsplat JIT-compiles its kernels, so it needs nvcc -- torch bundles only
    the CUDA runtime. Checked up front because otherwise the failure surfaces
    minutes later, deep inside the rasteriser, as an opaque AttributeError."""
    import shutil
    from torch.utils.cpp_extension import CUDA_HOME
    nvcc = shutil.which("nvcc")
    if not nvcc and CUDA_HOME:
        cand = os.path.join(CUDA_HOME, "bin", "nvcc")
        nvcc = cand if os.path.exists(cand) else None
    if not nvcc:
        sys.exit(
            "No CUDA compiler (nvcc) on PATH -- gsplat cannot build its kernels.\n\n"
            "  sudo apt-get -y install cuda-nvcc-12-4 cuda-cudart-dev-12-4 cuda-cccl-12-4\n\n"
            "then open a NEW shell and run `mapa` (it exports CUDA_HOME).\n"
            f"currently: CUDA_HOME={CUDA_HOME}, PATH has no nvcc"
        )
    return nvcc


def read_ply(path):
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.readline()
        arr = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    pts = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32)
    cols = np.stack([arr["red"], arr["green"], arr["blue"]], 1).astype(np.float32) / 255.0
    return pts, cols


def load_views(scene_dir, width, device):
    """Images, world-to-camera matrices and intrinsics at the requested width.

    Images are produced through MapAnything's own preprocessing so the crop
    matches the one its intrinsics were predicted for; the intrinsics then
    scale linearly with width (verified: a 2x render downsamples back onto the
    518px render to within resampling noise).
    """
    from mapanything.utils.image import load_images
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

    scene = json.load(open(os.path.join(scene_dir, "scene.json")))
    names = sorted(scene["frames"], key=lambda n: scene["frames"][n]["source"])
    if "R_c2w" not in scene["frames"][names[0]]:
        sys.exit("scene.json has no R_c2w ,  regenerate it with the current pipeline.")

    height = int(round(width * 294.0 / 518.0)) // 2 * 2
    paths = [scene["frames"][n]["source"] for n in names]

    def batched(ps, bs=12):
        for j in range(0, len(ps), bs):
            for v in load_images(ps[j:j + bs], resize_mode="fixed_size", size=(width, height)):
                yield v

    views = batched(paths)

    nrm = IMAGE_NORMALIZATION_DICT["dinov2"]
    mean = nrm.mean.view(1, 3, 1, 1)
    std = nrm.std.view(1, 3, 1, 1)

    imgs, viewmats, Ks = [], [], []
    Ks_base, s = [], None
    for n, v in zip(names, views):
        if s is None:
            act_h, act_w = v["img"].shape[-2:]
            if act_w != width:
                print(f"  note: preprocessing snapped {width}x{height} -> {act_w}x{act_h} "
                      f"(ViT patch size 14); using actual size")
            width, height = int(act_w), int(act_h)
            # Scale each axis independently. Preprocessing snaps BOTH dimensions
            # to the ViT patch size (14), which can change the aspect ratio --
            # 1890x1064 needs x3.6486 horizontally but x3.6190 vertically. Using
            # one factor for both leaves fy/cy ~0.8% off, a systematic vertical
            # misalignment that quietly degrades every rendered frame.
            s = (width / BASE_W, height / BASE_H)
            if abs(s[0] - s[1]) > 1e-6:
                print(f"  note: anisotropic scale x{s[0]:.4f}/y{s[1]:.4f}. Exactly "
                      f"aspect-preserving sizes are k*({BASE_W}x{BASE_H}): "
                      f"{', '.join(f'{BASE_W*k}x{BASE_H*k}' for k in (1,2,3,4))}")
        img = (v["img"] * std + mean).clamp(0, 1)[0]          # (3,H,W) in [0,1]
        imgs.append((img.permute(1, 2, 0) * 255).to(torch.uint8))   # (H,W,3) uint8

        fr = scene["frames"][n]
        Rc2w = np.array(fr["R_c2w"], dtype=np.float64)
        c = np.array(fr["centre"], dtype=np.float64)
        Rw2c = Rc2w.T
        vm = np.eye(4)
        vm[:3, :3] = Rw2c
        vm[:3, 3] = -Rw2c @ c
        viewmats.append(vm)

        K = np.load(os.path.join(scene_dir, f"frame_{n}.npz"))["intrinsics"][0].astype(np.float64)
        K = K.copy()
        K[0, :] *= s[0]
        K[1, :] *= s[1]
        Ks.append(K)
    assert imgs[0].shape[0] == height and imgs[0].shape[1] == width, "image/size mismatch"

    return (torch.stack(imgs),               # host-resident uint8: see note in main()
            torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device),
            torch.tensor(np.stack(Ks), dtype=torch.float32, device=device),
            width, height, names)


def init_gaussians(pts, cols, device, sh_degree, init_scale_k=3):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=init_scale_k + 1)
    spacing = np.clip(d[:, 1:].mean(1), 1e-4, None)            # mean dist to k neighbours

    N = len(pts)
    means = torch.tensor(pts, device=device)
    scales = torch.log(torch.tensor(spacing, dtype=torch.float32, device=device))[:, None].repeat(1, 3)
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
    opac = torch.logit(torch.full((N,), 0.1, device=device))
    sh0 = ((torch.tensor(cols, device=device) - 0.5) / SH_C0)[:, None, :]   # (N,1,3)
    shN = torch.zeros(N, (sh_degree + 1) ** 2 - 1, 3, device=device)

    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opac),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }).to(device)


def pose_delta_to_mat(delta):
    """(N,6) axis-angle + translation -> (N,4,4). Rodrigues; deltas stay small
    so the exact se(3) exponential's V term is not worth the arithmetic."""
    w, v = delta[:, :3], delta[:, 3:]
    th = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
    k = w / th
    K = torch.zeros(len(delta), 3, 3, device=delta.device, dtype=delta.dtype)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = torch.eye(3, device=delta.device, dtype=delta.dtype).expand(len(delta), 3, 3)
    th = th[:, :, None]
    R = I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    T = torch.eye(4, device=delta.device, dtype=delta.dtype).repeat(len(delta), 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = v
    return T


def ssim(a, b):
    """SSIM on (1,3,H,W) tensors with an 11x11 gaussian window."""
    g = torch.arange(11, dtype=torch.float32, device=a.device) - 5
    g = torch.exp(-(g ** 2) / (2 * 1.5 ** 2)); g = (g / g.sum())
    w = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()
    mu1 = F.conv2d(a, w, padding=5, groups=3)
    mu2 = F.conv2d(b, w, padding=5, groups=3)
    m1s, m2s, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(a * a, w, padding=5, groups=3) - m1s
    s2 = F.conv2d(b * b, w, padding=5, groups=3) - m2s
    s12 = F.conv2d(a * b, w, padding=5, groups=3) - m12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * m12 + C1) * (2 * s12 + C2)) / ((m1s + m2s + C1) * (s1 + s2 + C2))).mean()


def save_splat(params, path, compressed=False):
    """The 3DGS .ply convention stores RAW parameters -- log scales and logit
    opacity -- and the viewer applies exp()/sigmoid() itself. export_splats
    writes whatever it is handed, so passing activated values makes every
    viewer double-activate them: 100x-too-large splats and washed-out alpha.
    Pass the parameters exactly as trained."""
    export_splats(means=params["means"].detach(), scales=params["scales"].detach(),
                  quats=params["quats"].detach(), opacities=params["opacities"].detach(),
                  sh0=params["sh0"].detach(), shN=params["shN"].detach(),
                  format="ply_compressed" if compressed else "ply", save_to=path)


def main():
    ap = argparse.ArgumentParser(description="3DGS from a MapAnything scene")
    ap.add_argument("scene")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--width", type=int, default=1890,
                    help="training width. 1890 measured -3%% l1 vs 1554. Aspect-exact "
                         "sizes are k*(518x294); other sizes get per-axis intrinsics")
    ap.add_argument("--cap", type=int, default=1800000, help="max Gaussians (the 6GB knob)")
    ap.add_argument("--strategy", choices=["default", "mcmc"], default="default",
                    help="default = gradient-based densification. mcmc injects a random walk "
                         "into the means scaled by means_lr*noise_lr, which blows the scene "
                         "apart unless the LR is decayed on a schedule")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--sh-every", type=int, default=1000, help="raise active SH degree every N iters")
    ap.add_argument("--init-points", type=int, default=150000,
                    help="subsample init cloud (0 = all). 150k measured equal to a full "
                         "953k init and 2.4x faster; densification decides the real count")
    ap.add_argument("--opacity-reg", type=float, default=0.01)
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument("--preview-every", type=int, default=1000)
    ap.add_argument("--refine-poses", type=int, default=1,
                    help="jointly refine camera poses (1=on). Measured ~12px of residual "
                         "chunk drift across the track, which would otherwise bake in as blur")
    ap.add_argument("--pose-lr", type=float, default=1e-4)
    ap.add_argument("--grow-grad", type=float, default=5e-5,
                    help="DefaultStrategy densification threshold. Lower = more Gaussians "
                         "and THE dominant quality lever: 5e-5 vs the library default 2e-4 "
                         "gave 5x the Gaussians, -21%% mean L1, -18%% edge L1. --cap does not "
                         "bind at the library default, so this is what sets the real count")
    ap.add_argument("--compress", type=int, default=1,
                    help="also write splat_compressed.ply. Measured on a 1.5M splat: "
                         "5x smaller, 1.26mm mean position error (the input cloud is "
                         "voxelised at 20mm), and drops 23.5%% of Gaussians below "
                         "opacity 1/255. Good for sharing, keep the plain ply as master")
    ap.add_argument("--save-every", type=int, default=1000,
                    help="checkpoint splat.ply every N iters. Rasteriser memory grows "
                         "superlinearly in Gaussian count, so an OOM late in training is "
                         "a real risk on 6GB -- never lose the whole run to it")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required.")
    print("nvcc:", require_cuda_toolkit())
    dev = "cuda"
    os.makedirs(args.output, exist_ok=True)

    # Images stay on the CPU as uint8. Two separate ceilings force this:
    #   VRAM  - all-on-GPU makes VRAM scale with frame count (240 frames at
    #           1554x882 is 3.9 GB of images alone, on a 6 GB card).
    #   HOST  - as float32 on the CPU they are 16.4 MB each, and 80 of them
    #           plus a 4.1M-point cloud got this process OOM-killed inside
    #           WSL's 8 GB cap (anon-rss 7.45 GB, no swap). uint8 is 4x smaller
    #           and costs one cheap cast per iteration.
    imgs, viewmats, Ks, W, H, names = load_views(args.scene, args.width, dev)
    pts, cols = read_ply(os.path.join(args.scene, "points.ply"))
    if args.init_points and len(pts) > args.init_points:
        sel = np.random.default_rng(0).choice(len(pts), args.init_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    print(f"views {len(imgs)} at {W}x{H} | init points {len(pts):,} | cap {args.cap:,}")
    if args.cap <= len(pts):
        print(f"  WARNING: cap ({args.cap:,}) <= init points ({len(pts):,}). MCMC grows up to a\n"
              f"  cap, it does not prune down to one, so densification is effectively DISABLED\n"
              f"  and the result will stay soft. Raise --cap above the init count, or lower\n"
              f"  --init-points, so there is room to subdivide.", flush=True)

    params = init_gaussians(pts, cols, dev, args.sh_degree)
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    del pts, cols                      # the init cloud can be hundreds of MB
    lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": [params[k]], "lr": v, "name": k}], eps=1e-15)
            for k, v in lrs.items()}

    pose_delta = torch.nn.Parameter(torch.zeros(len(imgs), 6, device=dev))
    pose_opt = torch.optim.Adam([pose_delta], lr=args.pose_lr) if args.refine_poses else None

    if args.strategy == "mcmc":
        strat = MCMCStrategy(cap_max=args.cap, verbose=False)
    else:
        strat = DefaultStrategy(refine_stop_iter=int(args.iters * 0.75),
                                grow_grad2d=args.grow_grad, verbose=False)
    state = strat.initialize_state()

    # Standard 3DGS decays the means LR ~100x over training. Without it the
    # positions keep taking full-size steps forever -- and under MCMC the
    # injected noise scales with this same LR, which is what detonated run 2.
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / args.iters))

    n = len(imgs)
    order = np.random.default_rng(0).permutation(n)
    # Process wide counter, so reset it or a preceding stage's peak is reported here.
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for step in range(args.iters):
        i = int(order[step % n])
        if step % n == 0 and step:
            order = np.random.default_rng(step).permutation(n)

        sh_deg = min(args.sh_degree, step // args.sh_every)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        vm = viewmats[i:i+1]
        if pose_opt is not None:
            vm = pose_delta_to_mat(pose_delta[i:i+1]) @ vm

        render, _, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors, viewmats=vm, Ks=Ks[i:i+1],
            width=W, height=H, sh_degree=sh_deg, packed=True,
        )
        strat.step_pre_backward(params=params, optimizers=opts, state=state, step=step, info=info)

        gt = imgs[i:i+1].to(dev).permute(0, 3, 1, 2).float() / 255.0
        pr = render.permute(0, 3, 1, 2).clamp(0, 1)
        l1 = F.l1_loss(pr, gt)
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim(pr, gt))
        loss = loss + args.opacity_reg * torch.sigmoid(params["opacities"]).abs().mean() \
                    + args.scale_reg * torch.exp(params["scales"]).abs().mean()

        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        if pose_opt is not None:
            pose_opt.step(); pose_opt.zero_grad(set_to_none=True)
        if args.strategy == "mcmc":
            strat.step_post_backward(params=params, optimizers=opts, state=state,
                                     step=step, info=info, lr=sched.get_last_lr()[0])
        else:
            strat.step_post_backward(params=params, optimizers=opts, state=state,
                                     step=step, info=info, packed=True)
        sched.step()
        # DefaultStrategy has no hard cap; stop growth before VRAM does it for us
        if args.strategy == "default" and params["means"].shape[0] >= args.cap:
            strat.refine_stop_iter = min(strat.refine_stop_iter, step)

        if step % 100 == 0 or step == args.iters - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  {step:5d}/{args.iters}  loss {loss.item():.4f}  l1 {l1.item():.4f}  "
                  f"N {params['means'].shape[0]:,}  sh {sh_deg}  peak {mem:.2f} GB", flush=True)

        if args.save_every and step and step % args.save_every == 0:
            save_splat(params, os.path.join(args.output, "splat.ply"))

        if args.preview_every and step % args.preview_every == 0:
            import PIL.Image
            side = torch.cat([pr[0], gt[0]], 2).permute(1, 2, 0).detach().cpu().numpy()
            PIL.Image.fromarray((side * 255).astype(np.uint8)).save(
                os.path.join(args.output, f"preview_{step:05d}.jpg"), quality=90)

    dt = time.perf_counter() - t_start
    print(f"\ntraining: {dt/60:.1f} min for {args.iters} iters "
          f"({dt/args.iters*1000:.0f} ms/iter, {len(imgs)} views @ {W}x{H})")
    out = os.path.join(args.output, "splat.ply")
    save_splat(params, out)
    if args.compress:
        cpath = os.path.join(args.output, "splat_compressed.ply")
        save_splat(params, cpath, compressed=True)
        a, b = os.path.getsize(out), os.path.getsize(cpath)
        print(f"compressed copy: {b/1e6:.0f} MB vs {a/1e6:.0f} MB ({a/b:.1f}x smaller)")
    print(f"\n{params['means'].shape[0]:,} gaussians -> {out}")
    if pose_opt is not None:
        d = pose_delta.detach()
        print(f"pose refinement: max rotation {d[:, :3].norm(dim=1).max()*180/math.pi:.2f} deg, "
              f"max translation {d[:, 3:].norm(dim=1).max():.3f} m")
        np.save(os.path.join(args.output, "pose_delta.npy"), d.cpu().numpy())


if __name__ == "__main__":
    main()
