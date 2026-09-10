"""
ThriftSplat: one command from video to Gaussian splat.

    python -m thriftsplat clip.mp4 -o myscene

Runs both stages in order and writes everything under the output directory:

    myscene/scene/   metric point cloud, camera poses, per frame NPZ
    myscene/splat/   splat.ply, training previews, refined pose deltas

Run a single stage with --stage cloud or --stage splat. The splat stage reads
the cloud stage's output, so you can rerun training with different settings
without paying for reconstruction again.
"""

import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(
        prog="thriftsplat",
        description="Video to 3D Gaussian splat on a small GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("video", help="input video file")
    ap.add_argument("--output", "-o", required=True, help="output directory")
    ap.add_argument("--stage", choices=["all", "cloud", "splat"], default="all")

    g = ap.add_argument_group("reconstruction (stage 1)")
    g.add_argument("--fps", type=float, default=2.0, help="frames extracted per second")
    g.add_argument("--views", type=int, default=4, help="views per forward pass")
    g.add_argument("--size", type=int, default=0, help="longest side, 0 for native 518")
    g.add_argument("--overlap", type=int, default=2, help="frames shared between chunks")
    g.add_argument("--drop-blurry", type=float, default=0.15, help="fraction of blurriest frames dropped")
    g.add_argument("--voxel", type=float, default=0.02, help="point cloud merge size in metres")
    g.add_argument("--max-frames", type=int, default=0, help="cap total frames, 0 for no cap")

    s = ap.add_argument_group("splatting (stage 2)")
    s.add_argument("--iters", type=int, default=10000, help="training iterations")
    s.add_argument("--width", type=int, default=1890, help="training width in pixels")
    s.add_argument("--cap", type=int, default=1800000, help="maximum Gaussians")
    s.add_argument("--grow-grad", type=float, default=5e-5, help="densification threshold, lower grows more")
    s.add_argument("--init-points", type=int, default=150000, help="init cloud subsample, 0 for all")

    args = ap.parse_args()

    out = os.path.abspath(args.output)
    scene_dir = os.path.join(out, "scene")
    splat_dir = os.path.join(out, "splat")
    os.makedirs(out, exist_ok=True)

    t0 = time.perf_counter()

    if args.stage in ("all", "cloud"):
        if not os.path.exists(args.video):
            sys.exit(f"video not found: {args.video}")
        print("=" * 62)
        print("STAGE 1 of 2: video to metric point cloud")
        print("=" * 62, flush=True)
        from . import reconstruct
        sys.argv = [
            "reconstruct", args.video, "-o", scene_dir,
            "--fps", str(args.fps), "--views", str(args.views),
            "--size", str(args.size), "--overlap", str(args.overlap),
            "--drop-blurry", str(args.drop_blurry), "--voxel", str(args.voxel),
            "--max-frames", str(args.max_frames), "--rigid",
        ]
        reconstruct.main()
        t_cloud = time.perf_counter() - t0

    if args.stage in ("all", "splat"):
        if not os.path.exists(os.path.join(scene_dir, "scene.json")):
            sys.exit(f"no reconstruction at {scene_dir}. Run --stage cloud first.")
        print()
        print("=" * 62)
        print("STAGE 2 of 2: point cloud to Gaussian splat")
        print("=" * 62, flush=True)
        from . import train_gsplat
        sys.argv = [
            "train_gsplat", scene_dir, "-o", splat_dir,
            "--iters", str(args.iters), "--width", str(args.width),
            "--cap", str(args.cap), "--grow-grad", str(args.grow_grad),
            "--init-points", str(args.init_points),
            "--preview-every", "2500", "--save-every", "1000",
        ]
        train_gsplat.main()

    total = time.perf_counter() - t0
    print()
    print("=" * 62)
    if args.stage == "all":
        print(f"done in {total/60:.1f} min  (reconstruction {t_cloud/60:.1f} min, "
              f"splatting {(total-t_cloud)/60:.1f} min)")
    else:
        print(f"done in {total/60:.1f} min")
    print(f"  point cloud : {scene_dir}/points.ply")
    print(f"  splat       : {splat_dir}/splat.ply")
    print(f"  open the splat in SuperSplat, PlayCanvas, or a Blender 3DGS addon")
    print("=" * 62)


if __name__ == "__main__":
    main()
