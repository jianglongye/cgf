import argparse
import os
import sys
import time
from typing import Literal

import numpy as np
import torch
import trimesh
import viser

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cgf.data import YCB_CLASSES
from cgf.robotics import KinematicsLayer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
END_LINKS = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]  # end links for the allegro hand
URDF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data/original_robot/allegro_hand_ros/allegro_hand_description/"
)
DENSE_GEO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets/geometry/allegro_hand/right_dense_geo.npz")


def main(
    data_root: str,
    exp_dir: str,
    mano_side: Literal["left", "right"] = "left",
    obj_idx: int = 0,
):
    sample_dir = os.path.join(exp_dir, "sample")
    assert os.path.exists(sample_dir), f"sample_dir {sample_dir} does not exist"

    result = np.load(os.path.join(sample_dir, "result.npy"))
    # code = np.load(os.path.join(sample_dir, "code.npy"))

    # we hardcode the selected YCB object IDs
    selected_ycb_ids = [1, 3, 4, 5, 9, 10, 11, 12, 14, 15, 19, 21]
    ycb_id = selected_ycb_ids[obj_idx]
    ycb_model_path = os.path.join(data_root, "processed", "models", YCB_CLASSES[ycb_id], "textured_simple.obj")
    ycb_model = trimesh.load(ycb_model_path)

    # allegro hand
    if mano_side == "left":
        urdf_path = os.path.join(URDF_DIR, "allegro_hand_description_left.urdf")
    else:
        urdf_path = os.path.join(URDF_DIR, "allegro_hand_description_right.urdf")

    kinematics_layer = KinematicsLayer(
        urdf_path,
        END_LINKS,
        global_transform=True,
        geometry_path=DENSE_GEO_PATH,
        return_geometry=True,
        dtype=DTYPE,
        device=DEVICE,
    )

    server = viser.ViserServer()
    sample_slider = server.add_gui_slider("sample_idx", min=0, max=result.shape[1] - 1, step=1, initial_value=0)
    frame_slider = server.add_gui_slider("frame", min=0, max=result.shape[2] - 1, step=1, initial_value=0)
    server.add_mesh_trimesh("ycb", ycb_model)

    prev_frame = -1
    prev_sample_idx = -1
    while True:
        if sample_slider.value != prev_sample_idx:
            prev_sample_idx = sample_slider.value
            prev_frame = -1  # force update
            qpos = result[obj_idx, prev_sample_idx]
            tf3ds, geos = kinematics_layer(torch.from_numpy(qpos).to(DEVICE, dtype=DTYPE))
        if frame_slider.value != prev_frame:
            prev_frame = frame_slider.value
            for link_name in geos:
                server.add_mesh_simple(
                    f"allegro/{link_name}",
                    geos[link_name].verts_list()[prev_frame].cpu().numpy(),
                    geos[link_name].faces_list()[prev_frame].cpu().numpy(),
                    wireframe=True,
                )
        time.sleep(0.02)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--mano_side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--ts", type=str, default=None)
    args = parser.parse_args()
    assert os.path.exists(os.path.join(args.output_dir, args.ts)), f"exp_dir {args.output_dir}/{args.ts} does not exist"
    main(args.data_root, exp_dir=os.path.join(args.output_dir, args.ts), mano_side=args.mano_side)
