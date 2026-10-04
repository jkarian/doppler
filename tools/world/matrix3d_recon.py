"""Matrix-3D's panoramic-video-to-3D step (code/panoramic_video_to_3DScene.py) with its sizes as options: any video
size (e.g. one upscaled 2x by SeedVR2), no StableSR pass (the video is already upscaled, consistently), and a longer
splat training (upstream stops at 3,000 iterations; standard 3DGS uses ~30,000). Run from /opt/world/Matrix-3D.
"""

import argparse
import os
import shutil
import subprocess
import sys


def run(cmd: str) -> None:
    print("+", cmd[:160], flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inout_dir", required=True)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--iterations", type=int, default=15000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train_only", action="store_true", help="reuse the depth and views already in geom_optim/data")
    a = ap.parse_args()
    d = os.path.abspath(a.inout_dir)
    cond, gen, geo = f"{d}/condition", f"{d}/generated", f"{d}/geom_optim"
    if a.train_only:
        print("training only: reusing", f"{geo}/data", flush=True)
    else:
        run(f"{sys.executable} code/utils_3dscene/panorama_video_to_perspective_depth_sequential.py --device {a.device} "
            f"--camera_path {cond}/cameras.npz --video_path {gen}/generated.mp4 "
            f"--anchor_frame_depth_paths '{cond}/firstframe_depth.exr' --anchor_frame_mask_paths '{cond}/firstframe_mask.png' "
            f"--anchor_frame_indices 0 --output_dir {geo} --depth_estimation_interval 10 --width {a.width} --height {a.height}")
        run(f"{sys.executable} code/utils_3dscene/gs_optim_datagen.py --optimized_depth_dir {geo}/data/optimized_depths "
            f"--camera_path {cond}/cameras.npz --output_dir {geo}/data")
    it = a.iterations
    saves = " ".join(str(x) for x in sorted({3000, it // 2, it}))
    run(f"cd code/Pano_GS_Opt && {sys.executable} train.py -s {geo}/data -m {geo}/output -r 1 --use_decoupled_appearance "
        f"--save_iterations {saves} --test_iterations {it} --sh_degree 0 --densify_from_iter 500 "
        f"--densify_until_iter {max(1501, it // 2)} --iterations {it} --eval --img_sample_interval 1 --num_views_per_view 3 "
        f"--num_of_point_cloud 3000000 --device {a.device} --distortion_from_iter {it + 1} --depth_normal_from_iter {it + 1}")
    shutil.copy(f"{geo}/output/point_cloud/iteration_{it}/point_cloud.ply", f"{d}/generated_3dgs_opt.ply")
    print("wrote", f"{d}/generated_3dgs_opt.ply")


if __name__ == "__main__":
    main()
