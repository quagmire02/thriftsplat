#!/usr/bin/env python
"""
video -> MapAnything reconstruction, in one command.

    python pipeline/reconstruct.py drone.mp4 --output scene/

Stages: probe -> extract frames -> drop blurry frames -> chunk within the GPU's
pixel budget -> MapAnything per chunk -> align chunks into one world frame ->
write every per-frame output, the camera track, and a merged point cloud.

Chunking is not a tuning knob on this machine, it is forced: a 6 GB card fits
~700k pixels per forward pass (see lowvram.py). Chunks therefore overlap, and
consecutive chunks are stitched with a similarity transform fitted on the
cameras they share -- monocular reconstruction is only defined up to scale, so
each chunk comes back in its own gauge.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import torch

from . import lowvram

# Every per-view field MapAnything returns. All of it gets saved.
FIELDS = [
    "pts3d", "pts3d_cam", "depth_z", "depth_along_ray", "ray_directions",
    "intrinsics", "camera_poses", "cam_trans", "cam_quats", "conf", "mask",
    "non_ambiguous_mask", "metric_scaling_factor",
]


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames:format=duration",
         "-of", "json", video],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "width": s["width"], "height": s["height"],
        "fps": float(num) / float(den),
        "duration": float(d["format"]["duration"]),
    }


def extract_frames(video, outdir, fps):
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(outdir, f))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={fps}",
         "-q:v", "2", os.path.join(outdir, "f%05d.jpg")], check=True)
    return sorted(os.path.join(outdir, f) for f in os.listdir(outdir) if f.endswith(".jpg"))


def sharpness(paths):
    """Variance of the Laplacian. Motion-blurred frames score low, and a blurry
    frame in a chunk costs more than a missing one -- it corrupts the shared
    geometry the whole chunk is fitted against."""
    import cv2
    out = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        out.append(float(cv2.Laplacian(im, cv2.CV_64F).var()))
    return np.array(out)


def drop_blurry(paths, frac):
    if frac <= 0 or len(paths) < 10:
        return paths, np.array([])
    s = sharpness(paths)
    cutoff = np.quantile(s, frac)
    keep = s > cutoff
    keep[0] = keep[-1] = True  # endpoints anchor the track
    return [p for p, k in zip(paths, keep) if k], s


def chunks_of(paths, size, overlap):
    step = max(1, size - overlap)
    out = []
    i = 0
    while i < len(paths):
        c = paths[i:i + size]
        if len(c) < 2:
            if out:
                out[-1] = paths[max(0, len(paths) - size):]
            break
        out.append(c)
        if i + size >= len(paths):
            break
        i += step
    return out


def umeyama(X, Y, fixed_scale=False):
    """similarity s,R,t minimising ||sRX + t - Y|| for Nx3 point sets.

    fixed_scale forces s=1. MapAnything predicts metric geometry, so chunks
    should already share a scale; leaving scale free lets a single bad chunk
    fit a collapsing factor (0.155 observed) and drag the whole track with it."""
    n = X.shape[0]
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd((Yc.T @ Xc) / n)
    F = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        F[-1, -1] = -1
    R = U @ F @ Vt
    var = (Xc ** 2).sum() / n
    s = 1.0 if fixed_scale else (np.trace(np.diag(D) @ F) / var if var > 1e-12 else 1.0)
    return s, R, my - s * R @ mx


# MapAnything emits OpenCV world axes: +X right, +Y DOWN, +Z forward. Most
# viewers and DCC tools assume +Y up, so the scene loads upside down. This is a
# 180-degree rotation about X (det = +1), NOT a Y negation -- negating a single
# axis is a reflection and would mirror the scene and flip pose handedness.
Y_UP = np.diag([1.0, -1.0, -1.0])


def align_chunk(local_R, local_c, world_R, world_c, fixed_scale):
    """Fit s,R,t taking a chunk's local cameras onto their known world poses.

    Uses camera ORIENTATIONS, not just centres. Centre-only alignment is
    degenerate whenever the camera moves in a near-straight line: the shared
    centres are close to collinear, which leaves roll about the motion axis
    unconstrained, and chunks come back correctly placed but randomly rolled
    (a radial fan instead of a room). Orientations pin all three axes down.
    """
    M = sum(Rw @ Rl.T for Rw, Rl in zip(world_R, local_R))
    U, _, Vt = np.linalg.svd(M)
    F = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        F[-1, -1] = -1
    R = U @ F @ Vt
    if fixed_scale:
        s = 1.0
    else:
        lc = local_c - local_c.mean(0)
        wc = world_c - world_c.mean(0)
        s = float(np.sqrt((wc ** 2).sum() / max((lc ** 2).sum(), 1e-12)))
    t = world_c.mean(0) - s * R @ local_c.mean(0)
    return s, R, t


def to_np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else x


def main():
    ap = argparse.ArgumentParser(description="video -> MapAnything scene")
    ap.add_argument("video")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--fps", type=float, default=2.0, help="frames extracted per second")
    ap.add_argument("--views", type=int, default=10, help="views per chunk")
    ap.add_argument("--size", type=int, default=336, help="longest side; 0 = native 518 mapping")
    ap.add_argument("--overlap", type=int, default=2,
                    help="shared frames between chunks. 2 is enough because alignment uses "
                         "camera orientations, and measured faster AND tighter than 3 "
                         "(18 vs 34 chunks, 4.9mm vs 6.3mm stitch residual)")
    ap.add_argument("--drop-blurry", type=float, default=0.15, help="fraction of blurriest frames to discard")
    ap.add_argument("--max-frames", type=int, default=0, help="cap total frames (0 = no cap)")
    ap.add_argument("--voxel", type=float, default=0.02, help="point cloud voxel size in metres (0 = keep all)")
    ap.add_argument("--y-up", action="store_true",
                    help="rotate output 180 deg about X so +Y is up (MapAnything is +Y down)")
    ap.add_argument("--rigid", action="store_true",
                    help="stitch chunks without rescaling (MapAnything is metric, so scale should be 1)")
    args = ap.parse_args()

    if args.overlap < 2:
        print(f"warning: overlap={args.overlap} cannot constrain a chunk-to-chunk fit; use 2+")
    if args.overlap >= args.views:
        sys.exit(f"error: overlap ({args.overlap}) must be < views ({args.views})")

    os.makedirs(args.output, exist_ok=True)
    frames_dir = os.path.join(args.output, "frames")

    T = {}
    t0 = time.perf_counter()
    info = probe(args.video)
    print(f"video   : {info['width']}x{info['height']} @ {info['fps']:.2f} fps, {info['duration']:.1f}s")

    T["probe"] = time.perf_counter() - t0; t0 = time.perf_counter()
    paths = extract_frames(args.video, frames_dir, args.fps)
    T["extract"] = time.perf_counter() - t0; t0 = time.perf_counter()
    print(f"extracted: {len(paths)} frames at {args.fps} fps")

    paths, scores = drop_blurry(paths, args.drop_blurry)
    T["sharpness"] = time.perf_counter() - t0; t0 = time.perf_counter()
    if len(scores):
        print(f"sharpness: dropped {args.drop_blurry:.0%} blurriest -> {len(paths)} frames kept")

    if args.max_frames and len(paths) > args.max_frames:
        sel = np.linspace(0, len(paths) - 1, args.max_frames).round().astype(int)
        paths = [paths[i] for i in sel]
        print(f"capped   : {len(paths)} frames")

    budget_n = lowvram.max_views(args.size or 518)
    if args.views > budget_n:
        print(f"warning: --views {args.views} exceeds the ~{lowvram.PIXEL_BUDGET:,}px budget "
              f"at size {args.size or 518} (max {budget_n}); expect OOM")

    groups = chunks_of(paths, args.views, args.overlap)
    print(f"chunking : {len(groups)} chunks of <={args.views} views, {args.overlap} shared\n")

    model = lowvram.load_model()
    T["model_load"] = time.perf_counter() - t0; t0 = time.perf_counter()

    W = Y_UP if args.y_up else np.eye(3)   # output-only; alignment stays in native axes
    world = {}          # frame path -> global camera centre
    acc = None          # running similarity into the global frame
    all_pts, all_cols = [], []
    per_frame = {}
    stitches = []
    resid = float("nan")

    for ci, group in enumerate(groups):
        preds = lowvram.infer(model, group, longest_side=args.size or None)
        poses = np.stack([to_np(p["camera_poses"])[0] for p in preds])
        centres = poses[:, :3, 3]
        rots = poses[:, :3, :3]

        if acc is None:
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            shared = [i for i, p in enumerate(group) if p in world]
            if len(shared) >= 2:
                tgt = np.stack([world[group[i]][0] for i in shared])
                tgtR = np.stack([world[group[i]][1] for i in shared])
                s, R, t = align_chunk(rots[shared], centres[shared], tgtR, tgt, args.rigid)
                resid = float(np.sqrt((np.linalg.norm(
                    (s * (R @ centres[shared].T)).T + t - tgt, axis=1) ** 2).mean()))
            else:
                s, R, t = acc
                resid = float("nan")
                print(f"  chunk {ci}: only {len(shared)} shared frames, reusing previous transform")
        acc = (s, R, t)

        gc = (s * (R @ centres.T)).T + t
        gr = np.einsum("ij,njk->nik", R, rots)
        for p, c, rr in zip(group, gc, gr):
            world.setdefault(p, (c, rr))

        for k, (p, pred) in enumerate(zip(group, preds)):
            name = os.path.splitext(os.path.basename(p))[0]
            if name in per_frame:
                continue
            rec = {f: to_np(pred[f]) for f in FIELDS if f in pred}
            pts = rec["pts3d"][0]
            wp = ((s * (R @ pts.reshape(-1, 3).T)).T + t) @ W.T
            rec["pts3d_world"] = wp.reshape(pts.shape)
            rec["chunk"] = ci
            np.savez_compressed(os.path.join(args.output, f"frame_{name}.npz"), **rec)
            per_frame[name] = {"chunk": ci, "centre": (W @ gc[k]).tolist(),
                               "R_c2w": (W @ gr[k]).tolist(), "source": p}

            m = rec["mask"][0].squeeze() > 0 if "mask" in rec else np.ones(pts.shape[:2], bool)
            m &= np.isfinite(rec["pts3d_world"]).all(-1)
            if m.sum():
                import cv2
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (pts.shape[1], pts.shape[0]))
                all_pts.append(rec["pts3d_world"][m])
                all_cols.append(img[m])

        flag = ""
        if ci and (s < 0.5 or s > 2.0):
            flag = "  <-- SCALE OUTLIER, stitch likely wrong"
        elif ci and resid == resid and resid > 0.5:
            flag = "  <-- high stitch residual"
        stitches.append({"chunk": ci, "scale": float(s), "residual_m": resid, "views": len(group)})
        print(f"  chunk {ci+1}/{len(groups)}: {len(group)} views, scale {s:.3f}, "
              f"stitch residual {resid:.3f} m{flag}" if ci else
              f"  chunk {ci+1}/{len(groups)}: {len(group)} views, reference chunk")

    T["inference"] = time.perf_counter() - t0; t0 = time.perf_counter()
    pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
    cols = np.concatenate(all_cols) if all_cols else np.zeros((0, 3), np.uint8)

    if args.voxel > 0 and len(pts):
        keys = np.floor(pts / args.voxel).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        pts, cols = pts[keep], cols[keep]

    write_ply(os.path.join(args.output, "points.ply"), pts, cols)
    json.dump({"video": os.path.abspath(args.video), "video_info": info,
               "args": vars(args), "chunks": len(groups), "stitches": stitches,
               "frames": per_frame},
              open(os.path.join(args.output, "scene.json"), "w"), indent=1)

    print(f"\npoints   : {len(pts):,} -> {args.output}/points.ply")
    print(f"per-frame: {len(per_frame)} .npz files with {len(FIELDS)} fields each")
    # Parallax diagnostic. Reconstruction quality tracks the PERPENDICULAR
    # baseline, not raw camera speed: motion along the view axis (dolly-in,
    # nadir descent) yields almost no parallax near image centre however far
    # you travel. Measured across three clips this rank-ordered reprojection
    # quality perfectly, where raw baseline/depth got the order backwards.
    try:
        nm = sorted(per_frame, key=lambda n: per_frame[n]["source"])
        Cc = np.array([per_frame[n]["centre"] for n in nm])
        Rr = np.array([per_frame[n]["R_c2w"] for n in nm])
        dep = []
        for n in nm[:: max(1, len(nm) // 12)]:
            z = np.load(os.path.join(args.output, f"frame_{n}.npz"))["depth_z"][0].squeeze()
            z = z[np.isfinite(z) & (z > 1e-3)]
            if len(z):
                dep.append(np.median(z))
        depth = float(np.median(dep)) if dep else float("nan")
        perp = []
        for i in range(len(Cc) - 1):
            v = Cc[i + 1] - Cc[i]
            nv = np.linalg.norm(v)
            if nv > 1e-6:
                fwd = abs(v @ Rr[i][:, 2]) / nv
                perp.append(nv * np.sqrt(max(0.0, 1 - fwd ** 2)))
        pb = float(np.median(perp)) / depth if perp and depth == depth else float("nan")
        verdict = ("good" if pb > 0.02 else "marginal" if pb > 0.012 else "LOW - expect soft geometry")
        print(f"\nparallax : perpendicular baseline/depth = {pb:.4f}  ({verdict})")
        print(f"           median depth {depth:.2f} m, sideways motion "
              f"{np.median(perp)*100:.1f} cm/frame")
        if pb <= 0.012:
            print("           camera moved mostly ALONG its view axis. Arc or strafe "
                  "instead of dollying straight in.")
    except Exception as e:
        print(f"\nparallax : n/a ({type(e).__name__})")

    # Release the reconstruction model. When both stages run in one process the
    # caching allocator otherwise holds ~2.5 GB of weights through training.
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    T["output"] = time.perf_counter() - t0
    print(f"metadata : {args.output}/scene.json")
    tot = sum(T.values())
    print(f"\ntiming ({tot/60:.1f} min total, {tot/max(len(per_frame),1):.1f} s/frame):")
    for k, v in T.items():
        print(f"  {k:11s} {v:7.1f}s  {v/tot*100:4.1f}%")


def write_ply(path, pts, cols):
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(pts)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")
        arr = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        arr["red"], arr["green"], arr["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
        f.write(arr.tobytes())


if __name__ == "__main__":
    main()
