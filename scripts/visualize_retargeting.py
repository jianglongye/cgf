import argparse
import json
import os
import sys
import time
import warnings
from typing import Literal
from xml.etree import ElementTree

import numpy as np
import torch
import viser
from manopth.manolayer import ManoLayer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cgf.robotics import KinematicsLayer, rescale_urdf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
END_LINKS = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]  # end links for the allegro hand
MANO_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/mano_v1_2_models")
URDF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data/original_robot/allegro_hand_ros/allegro_hand_description/"
)
DENSE_GEO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets/geometry/allegro_hand/right_dense_geo.npz")


def main(data_root: str, mano_side: Literal["left", "right"] = "left", seq_id: str = None, aug_idx: int = 0):
    # human hand
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mano_layer = ManoLayer(flat_hand_mean=False, ncomps=45, side=mano_side, mano_root=MANO_DIR, use_pca=True)
        mano_layer = mano_layer.to(DEVICE)

    with open(os.path.join(args.data_root, "processed", "meta.json"), "r") as f:
        meta = json.load(f)
    pose_m_aug_npz_file = np.load(os.path.join(data_root, "processed", "pose_m_aug.npz"))
    seq_ids = [x[:-5] for x in pose_m_aug_npz_file.files if "mano" in x]
    seq_ids = [x for x in seq_ids if meta[x]["mano_sides"][0] == args.mano_side]
    sorted_seq_ids = sorted(seq_ids)

    seq_betas = np.array([meta[seq_id]["betas"] for seq_id in sorted_seq_ids])
    valid_frames_range = np.array([meta[seq_id]["valid_frames_range"] for seq_id in sorted_seq_ids])
    # add 1 to the end frame
    valid_frames_range[:, 1] += 1

    if seq_id is None:
        seq_id = sorted_seq_ids[0]
    assert seq_id in sorted_seq_ids
    begin_f, end_f = valid_frames_range[sorted_seq_ids.index(seq_id)]

    mano_pose = pose_m_aug_npz_file[f"{seq_id}_mano"][:, begin_f:end_f]
    mano_pose = torch.from_numpy(mano_pose[aug_idx, :, 0]).clone().to(DEVICE, dtype=DTYPE)
    betas = (
        torch.from_numpy(seq_betas[sorted_seq_ids.index(seq_id)])
        .clone()
        .expand(mano_pose.shape[0], -1)
        .to(DEVICE, dtype=DTYPE)
    )
    mano_verts, mano_joint = mano_layer(mano_pose[..., :48], betas, mano_pose[..., 48:51])
    mano_verts /= 1000
    mano_verts = mano_verts.cpu().numpy()
    mano_faces = mano_layer.th_faces.cpu().numpy()

    # allegro hand
    if args.mano_side == "left":
        urdf_path = os.path.join(URDF_DIR, "allegro_hand_description_left.urdf")
    else:
        urdf_path = os.path.join(URDF_DIR, "allegro_hand_description_right.urdf")
    with open(urdf_path, "r") as f:
        urdf_data = ElementTree.parse(f)
    new_urdf_data = rescale_urdf(urdf_data, 0.8)
    new_urdf_str = ElementTree.tostring(new_urdf_data.getroot(), encoding="unicode")
    kinematics_layer = KinematicsLayer(
        new_urdf_str,
        END_LINKS,
        global_transform=True,
        geometry_path=DENSE_GEO_PATH,
        return_geometry=True,
        dtype=DTYPE,
        device=DEVICE,
    )
    qpos = np.load(os.path.join(data_root, "processed", f"retargeting_{mano_side}", f"{aug_idx}.npz"))["qpos"][aug_idx]

    tf3ds, geos = kinematics_layer(torch.from_numpy(qpos).to(DEVICE, dtype=DTYPE))

    server = viser.ViserServer()
    frame_slider = server.add_gui_slider("frame", min=0, max=mano_pose.shape[0] - 1, step=1, initial_value=0)

    prev_frame = -1
    while True:
        if frame_slider.value != prev_frame:
            prev_frame = frame_slider.value
            server.add_mesh_simple("hand", mano_verts[prev_frame], mano_faces, wireframe=True)
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
    parser.add_argument("--mano_side", type=str, choices=["left", "right"], default="left")
    args = parser.parse_args()
    main(args.data_root, args.mano_side)
