import argparse
import json
import os
import sys
import warnings
from xml.etree import ElementTree

import numpy as np
import torch
import transforms3d.euler
from manopth.manolayer import ManoLayer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cgf.robotics import KinematicsLayer, load_joint_limits, rescale_urdf
from cgf.transformation import axis_angle_to_matrix, matrix_to_axis_angle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

END_LINKS = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]  # end links for the allegro hand
MANO_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/mano_v1_2_models")
URDF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data/original_robot/allegro_hand_ros/allegro_hand_description/"
)
DENSE_GEO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets/geometry/allegro_hand/right_dense_geo.npz")


def compute_hand_geometry(pose_m, betas):
    p = torch.from_numpy(pose_m[:, :48]).to(DEVICE).to(DTYPE)
    t = torch.from_numpy(pose_m[:, 48:51]).to(DEVICE).to(DTYPE)
    betas = torch.from_numpy(betas[None]).to(DEVICE).to(DTYPE)

    vertex, joint = mano_layer(p, betas, t)

    vertex /= 1000
    joint /= 1000

    return vertex, joint


def batch_retargeting(pose_m_dataset, betas_dataset, num_valid_frames, target_mano_side="left"):
    link_hand_indices = [0, 4, 8, 12, 16]
    link_robot_names = ["palm", "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]

    loss_fn = torch.nn.SmoothL1Loss(beta=0.01)

    batch_size, max_frame_num, _ = pose_m_dataset.shape

    full_qpos = torch.zeros(batch_size, 22, device=DEVICE, dtype=DTYPE)
    full_qpos[:, 6:] = joint_limits.mean(1).unsqueeze(0).repeat(batch_size, 1)

    full_qpos[:, 0:3] = pose_m_dataset[:, 0, 48:51].clone()
    mano_rot_axisangle = pose_m_dataset[:, 0, 0:3].clone()
    mano_rot_mat = axis_angle_to_matrix(mano_rot_axisangle)
    right_mano2robot = torch.from_numpy(transforms3d.euler.euler2mat(np.pi / 2, 0, -np.pi / 2))
    left_mano2robot = torch.from_numpy(transforms3d.euler.euler2mat(-np.pi / 2, 0, -np.pi / 2))
    mano2robot = left_mano2robot if target_mano_side == "left" else right_mano2robot
    mano2robot = mano2robot.unsqueeze(0).repeat(batch_size, 1, 1).to(DTYPE).to(DEVICE)
    full_qpos[:, 3:6] = matrix_to_axis_angle(torch.bmm(mano_rot_mat, mano2robot))
    full_qpos = full_qpos.clone()

    result = np.zeros(pose_m_dataset.shape[:2] + (22,))
    for frame_idx in range(max_frame_num):
        mask = frame_idx < num_valid_frames
        print(f"frame: {frame_idx}, valid num: {torch.count_nonzero(mask).item()}")
        mano_pose = pose_m_dataset[mask, frame_idx]
        betas = betas_dataset[mask]

        mano_verts, mano_joint = mano_layer(mano_pose[:, :48], betas, mano_pose[:, 48:51])

        mano_verts /= 1000
        mano_joint /= 1000

        cur_qpos = full_qpos[mask].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([cur_qpos], lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 300], gamma=0.8)

        target_pos = mano_joint[:, link_hand_indices]

        for iter in range(60 if frame_idx == 0 else 30):
            optimizer.zero_grad()
            tf3ds = kinematics_layer(cur_qpos)
            end_pos = torch.stack([tf3ds[name].get_matrix()[:, :3, 3] for name in link_robot_names]).permute(1, 0, 2)
            assert link_robot_names[0] == "palm"
            assert end_pos.shape == target_pos.shape
            pos_loss = loss_fn(end_pos, target_pos)
            # i_loss = self_intersection_loss(geos, groups, other_groups) * 100
            loss = pos_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"iter: {iter}, loss: {pos_loss.item():.6f}")

        full_qpos[mask] = cur_qpos.detach()
        result[mask.cpu().numpy(), frame_idx] = cur_qpos.detach().cpu().numpy()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--mano_side", type=str, choices=["left", "right"], default="left")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mano_layer = ManoLayer(flat_hand_mean=False, ncomps=45, side=args.mano_side, mano_root=MANO_DIR, use_pca=True)
        mano_layer = mano_layer.to(DEVICE)

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
        return_geometry=False,
        dtype=DTYPE,
        device=DEVICE,
    )
    joint_limits = load_joint_limits(urdf_path)
    joint_limits = torch.tensor(list(joint_limits.values()), device=DEVICE, dtype=DTYPE)

    groups = [
        ["link_3.0", "link_3.0_tip"],
        ["link_7.0", "link_7.0_tip"],
        ["link_11.0", "link_11.0_tip"],
        ["link_15.0", "link_15.0_tip"],
        ["base_link"],
    ]

    other_groups = []
    for i in range(len(groups)):
        other_groups.append([x for j in range(len(groups)) if i != j for x in groups[j]])

        with open(os.path.join(args.data_root, "processed", "meta.json"), "r") as f:
            meta = json.load(f)
        pose_m_aug_npz_file = np.load(os.path.join(args.data_root, "processed", "pose_m_aug.npz"))

        seq_ids = [x[:-5] for x in pose_m_aug_npz_file.files if "mano" in x]
        seq_ids = [x for x in seq_ids if meta[x]["mano_sides"][0] == args.mano_side]
        sorted_seq_ids = sorted(seq_ids)
        print(f"len(sorted_seq_ids): {len(sorted_seq_ids)}")

    output_dir = os.path.join(args.data_root, "processed", f"retargeting_{args.mano_side}")
    os.makedirs(output_dir, exist_ok=True)

    valid_frames_range = np.array([meta[seq_id]["valid_frames_range"] for seq_id in sorted_seq_ids])
    # add 1 to the end frame
    valid_frames_range[:, 1] += 1
    num_valid_frames = valid_frames_range[:, 1] - valid_frames_range[:, 0]
    max_frame_num = int(np.max(num_valid_frames))

    seq_aug_num = 11
    pose_m_dataset = np.zeros((len(sorted_seq_ids), seq_aug_num, max_frame_num, 51))
    for idx, seq_id in enumerate(sorted_seq_ids):
        begin_f, end_f = valid_frames_range[idx]
        pose_m_dataset[idx, :, : end_f - begin_f] = pose_m_aug_npz_file[f"{seq_id}_mano"][:, begin_f:end_f, 0]

    betas_dataset = np.array([meta[seq_id]["betas"] for seq_id in sorted_seq_ids])
    aug_rots = np.array([pose_m_aug_npz_file[f"{seq_id}_aug_rot"] for seq_id in sorted_seq_ids])

    pose_m_dataset = torch.from_numpy(pose_m_dataset).to(DTYPE).to(DEVICE)
    betas_dataset = torch.from_numpy(betas_dataset).to(DTYPE).to(DEVICE)
    num_valid_frames = torch.from_numpy(num_valid_frames).to(DTYPE).to(DEVICE)

    for aug_idx in range(0, 11):
        print(f"aug_idx: {aug_idx}")
        result = batch_retargeting(pose_m_dataset[:, aug_idx], betas_dataset, num_valid_frames, args.mano_side)
        np.savez_compressed(os.path.join(output_dir, f"{aug_idx}.npz"), qpos=result, aug_rot=aug_rots[:, aug_idx])
