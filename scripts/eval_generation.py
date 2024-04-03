import argparse
import os
import sys

import numpy as np
import torch
import trimesh
from einops import rearrange
from pytorch3d.ops import knn_points

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cgf.data import YCB_CLASSES
from cgf.robotics import KinematicsLayer
from cgf.transformation import euler_angles_to_matrix, matrix_to_rotation_6d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

END_LINKS = ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"]
URDF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data/original_robot/allegro_hand_ros/allegro_hand_description/"
)
DENSE_GEO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets/geometry/allegro_hand/right_dense_geo.npz")


def eval_smoothness(data, total_spacing=1):
    if isinstance(data, np.ndarray):
        rot_6d = matrix_to_rotation_6d(euler_angles_to_matrix(data[..., 3:6], convention="rxyz"))
        data = np.concatenate([data[..., :3], data[..., 6:]], axis=-1)
        data = (data - data[:, 0:1, :]) / (data[:, -1:, :] - data[:, 0:1, :])
        spacing = total_spacing / data.shape[1]
        vel = np.gradient(data, spacing, axis=1)
        acc = np.gradient(vel, spacing, axis=1)
        pos_std = np.abs(data[:, :-1, :] - data[:, 1:, :]).sum(axis=1).mean()
        vel_std = np.abs(vel[:, :-1, :] - vel[:, 1:, :]).sum(axis=1).mean()
        acc_std = np.abs(acc[:, :-1, :] - acc[:, 1:, :]).sum(axis=1).mean()
    else:
        rot_6d = [matrix_to_rotation_6d(euler_angles_to_matrix(d[..., 3:6], convention="rxyz")) for d in data]
        data = [np.concatenate([d[..., :3], d[..., 6:]], axis=-1) for d, r in zip(data, rot_6d)]
        data = [(d - d[0:1, :]) / (d[-1:, :] - d[0:1, :]) for d in data]
        vel = [np.gradient(d, total_spacing / d.shape[0], axis=0) for d in data]
        acc = [np.gradient(d, total_spacing / d.shape[0], axis=0) for d in vel]
        pos_std = np.array([np.abs(d[:-1, :] - d[1:, :]).sum(axis=0).mean() for d in data]).mean()
        vel_std = np.array([np.abs(d[:-1, :] - d[1:, :]).sum(axis=0).mean() for d in vel]).mean()
        acc_std = np.array([np.abs(d[:-1, :] - d[1:, :]).sum(axis=0).mean() for d in acc]).mean()
    return pos_std, vel_std, acc_std


@torch.no_grad()
def eval_contact(data_root, data, batch_size=2000, threshold=5e-3):
    obj_pts_path = os.path.join(data_root, "processed", "models", ycb_name, "points.xyz")
    obj_pts = np.asarray(trimesh.load(obj_pts_path).vertices)
    obj_pts = torch.from_numpy(obj_pts).float().to(DEVICE)

    if data.ndim == 3:
        last_frame_data = data[:, -1]
    else:
        last_frame_data = data
    last_frame_data = torch.from_numpy(last_frame_data).to(device=DEVICE, dtype=DTYPE)

    contact_ratio, obj_verts_ratio, hand_verts_ratio, finger_num = [], [], [], []

    begin, end = 0, min(batch_size, last_frame_data.shape[0])
    while begin < last_frame_data.shape[0]:
        batch_data = last_frame_data[begin:end]
        tf3ds, geos = kinematics_layer(batch_data)

        src_xyz = torch.cat([geos[link_name].verts_padded() for link_name in geos], dim=1)
        trg_xyz = obj_pts.unsqueeze(0).expand(batch_data.shape[0], -1, -1)

        # [B], N for each num
        src_lengths = torch.full((src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device)
        trg_lengths = torch.full((trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device)

        src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=1)
        # NOTE that torch.sqrt is necessary to get the actual distance
        src_nn_dists, src_nn_idx = torch.sqrt(src_nn.dists[..., 0]), src_nn.idx[..., 0]

        trg_nn = knn_points(trg_xyz, src_xyz, lengths1=trg_lengths, lengths2=src_lengths, K=1)
        trg_nn_dists, trg_nn_idx = torch.sqrt(trg_nn.dists[..., 0]), trg_nn.idx[..., 0]

        contact_ratio.append(torch.any(src_nn_dists < threshold, dim=1))
        obj_verts_ratio.append(torch.sum(trg_nn_dists < threshold, dim=1) / trg_nn_dists.shape[1])
        hand_verts_ratio.append(torch.sum(src_nn_dists < threshold, dim=1) / src_nn_dists.shape[1])

        end_link_contact_num = torch.zeros(batch_data.shape[0], device=DEVICE, dtype=DTYPE)
        for i in range(len(END_LINKS)):
            xyz = geos[END_LINKS[i]].verts_padded()
            lengths = torch.full((xyz.shape[0],), xyz.shape[1], dtype=torch.int64, device=xyz.device)
            nn_dists = knn_points(xyz, trg_xyz, lengths1=lengths, lengths2=trg_lengths, K=1).dists[..., 0]
            end_link_contact_num += torch.any(torch.sqrt(nn_dists) < threshold, dim=1).to(DTYPE)
        finger_num.append(end_link_contact_num)

        begin, end = end, min(end + batch_size, last_frame_data.shape[0])

    contact_ratio = torch.cat(contact_ratio, dim=0).sum() / last_frame_data.shape[0]
    obj_verts_ratio = torch.cat(obj_verts_ratio, dim=0).mean()
    hand_verts_ratio = torch.cat(hand_verts_ratio, dim=0).mean()
    finger_num = torch.cat(finger_num, dim=0).mean()

    return contact_ratio, obj_verts_ratio, hand_verts_ratio, finger_num


def eval_joint_smoothness(data, total_spacing=1):
    if isinstance(data, np.ndarray):
        data = (data - data[:, 0:1, :]) / (data[:, -1:, :] - data[:, 0:1, :])
        spacing = total_spacing / data.shape[1]
        vel = np.gradient(data, spacing, axis=1)
        acc = np.gradient(vel, spacing, axis=1)
        pos_std = np.abs(data[:, :-1, :] - data[:, 1:, :]).sum(axis=1).mean()
        vel_std = np.abs(vel[:, :-1, :] - vel[:, 1:, :]).sum(axis=1).mean()
        acc_std = np.abs(acc[:, :-1, :] - acc[:, 1:, :]).sum(axis=1).mean()
    else:
        data = [(d - d[0:1, :]) / (d[-1:, :] - d[0:1, :]) for d in data]
        vel = [np.gradient(d, total_spacing / d.shape[0], axis=0) for d in data]
        acc = [np.gradient(d, total_spacing / d.shape[0], axis=0) for d in vel]
        pos_std = np.array([np.abs(d[:-1, :] - d[1:, :]).sum(axis=0).mean() for d in data]).mean()
        vel_std = np.array([np.abs(d[:-1, :] - d[1:, :]).sum(axis=0).mean() for d in vel]).mean()
        acc_std = np.array([np.abs(d[:-1, :] - d[1:, :]).sum(axis=0).mean() for d in acc]).mean()
    return pos_std, vel_std, acc_std


@torch.no_grad()
def qpos_to_joints(data, batch_size=4000):
    if isinstance(data, np.ndarray):
        assert data.ndim == 3
        ori_shape = data.shape
        data = torch.from_numpy(data).to(dtype=DTYPE, device=DEVICE)
        data = rearrange(data, "b n d -> (b n) d")

        joints = []
        begin, end = 0, min(batch_size, data.shape[0])
        while begin < data.shape[0]:
            batch_data = data[begin:end]
            tf3ds = kinematics_layer_no_geo(batch_data)
            batch_joints = torch.stack([tf3ds[link_name].get_matrix()[..., :3, 3] for link_name in tf3ds], dim=1)
            joints.append(batch_joints.cpu().numpy())
            begin, end = end, min(end + batch_size, data.shape[0])

        joints = np.concatenate(joints, axis=0)
        joints = rearrange(joints, "(b n) j d -> b n j d", b=ori_shape[0], n=ori_shape[1])
        return joints
    elif isinstance(data, list):
        lengths = [x.shape[0] for x in data]
        data = np.concatenate([x for x in data], axis=0)
        data = torch.from_numpy(data).to(dtype=DTYPE, device=DEVICE)
        joints = []
        begin, end = 0, min(batch_size, data.shape[0])
        while begin < data.shape[0]:
            batch_data = data[begin:end]
            tf3ds = kinematics_layer_no_geo(batch_data)
            batch_joints = torch.stack([tf3ds[link_name].get_matrix()[..., :3, 3] for link_name in tf3ds], dim=1)
            joints.append(batch_joints.cpu().numpy())
            begin, end = end, min(end + batch_size, data.shape[0])

        joints = np.concatenate(joints, axis=0)
        ends = np.cumsum(lengths)
        new_joints = [joints[ends[i] - lengths[i] : ends[i]] for i in range(len(lengths))]
        return new_joints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--mano_side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--ts", type=str, default=None)
    args = parser.parse_args()
    assert os.path.exists(os.path.join(args.output_dir, args.ts)), f"exp_dir {args.output_dir}/{args.ts} does not exist"

    # allegro hand
    if args.mano_side == "left":
        urdf_path = os.path.join(URDF_DIR, "allegro_hand_description_left.urdf")
    else:
        urdf_path = os.path.join(URDF_DIR, "allegro_hand_description_right.urdf")

    kinematics_layer_no_geo = KinematicsLayer(
        urdf_path,
        END_LINKS,
        global_transform=True,
        return_geometry=False,
        device=DEVICE,
        dtype=DTYPE,
    )
    kinematics_layer = KinematicsLayer(
        urdf_path,
        END_LINKS,
        global_transform=True,
        geometry_path=DENSE_GEO_PATH,
        return_geometry=True,
        device=DEVICE,
        dtype=DTYPE,
    )

    selected_ycb_ids = [1, 3, 4, 5, 9, 10, 11, 12, 14, 15, 19, 21]

    coord_pos_stds, coord_vel_stds, coord_acc_stds = [], [], []
    joints_pos_stds, joints_vel_stds, joints_acc_stds = [], [], []

    for ycb_id in selected_ycb_ids:
        ycb_name = YCB_CLASSES[ycb_id]
        exp_dir = os.path.join(args.output_dir, args.ts)
        result_path = os.path.join(exp_dir, "sample/result_filter_sapien", f"{ycb_name}.npy")

        result = np.load(result_path)

        linear = False
        if linear:
            step_size = (result[:, -1:] - result[:, 0:1]) / (result.shape[1] - 1)
            result[:, 1:] = result[:, 0:1] + np.arange(1, 40)[None, :, None] * step_size

        if not isinstance(result, np.ndarray):
            result = [result[k] for k in result]
            print(f"dist: {np.array([np.linalg.norm(x[-1, :3] - x[0, :3]) for x in result]).mean():.6f}")
        else:
            print(f"dist: {np.linalg.norm(result[:, -1, :3] - result[:, 0, :3], axis=-1).mean():.6f}")
        pos_std, vel_std, acc_std = eval_smoothness(result, total_spacing=1)
        coord_pos_stds.append(pos_std)
        coord_vel_stds.append(vel_std)
        coord_acc_stds.append(acc_std)
        print(f"pos_std: {pos_std:.6f}, vel_std: {vel_std:.6f}, acc_std: {acc_std:.6f}")

        joints = qpos_to_joints(result)
        pos_std, vel_std, acc_std = eval_joint_smoothness(joints, total_spacing=1)
        joints_pos_stds.append(pos_std)
        joints_vel_stds.append(vel_std)
        joints_acc_stds.append(acc_std)
        print("eval on joints")
        print(f"pos_std: {pos_std:.6f}, vel_std: {vel_std:.6f}, acc_std: {acc_std:.6f}")

    print(f"coord_pos_stds: {np.array(coord_pos_stds).mean():.6f}")
    print(f"coord_vel_stds: {np.array(coord_vel_stds).mean():.6f}")
    print(f"coord_vel_stds: {np.log10(np.array(coord_vel_stds).mean()):.6f} (log10)")
    print(f"coord_acc_stds: {np.array(coord_acc_stds).mean():.6f}")
    print(f"coord_acc_stds: {np.log10(np.array(coord_acc_stds).mean()):.6f} (log10)")
    print(f"joints_pos_stds: {np.array(joints_pos_stds).mean():.6f}")
    print(f"joints_vel_stds: {np.array(joints_vel_stds).mean():.6f}")
    print(f"joints_vel_stds: {np.log10(np.array(joints_vel_stds).mean()):.6f} (log10)")
    print(f"joints_acc_stds: {np.array(joints_acc_stds).mean():.6f}")
    print(f"joints_acc_stds: {np.log10(np.array(joints_acc_stds).mean()):.6f} (log10)")
