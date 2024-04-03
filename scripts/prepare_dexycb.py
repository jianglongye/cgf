import argparse
import json
import os
import shutil
import sys

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cgf.transformation import ManoTransformation


def extract_meta(data_root, moving_threshold=0.004):
    subject_ids = [x for x in os.listdir(data_root) if "subject" in x]

    subject_seq_list = []
    for subject_id in subject_ids:
        seq_ids = os.listdir(os.path.join(data_root, subject_id))
        subject_seq_list.extend([(subject_id, x) for x in seq_ids])

    subject_seq_list = sorted(subject_seq_list, key=lambda x: os.path.join(*x))

    results = {}
    for subject_id, seq_id in subject_seq_list:
        pose_path = os.path.join(data_root, subject_id, seq_id, "pose.npz")
        meta_path = os.path.join(data_root, subject_id, seq_id, "meta.yml")

        with open(meta_path, "r") as f:
            meta_item = yaml.safe_load(f)
        pose_npz = np.load(pose_path)

        mano_calib_path = os.path.join(data_root, "calibration", f"mano_{meta_item['mano_calib'][0]}", "mano.yml")
        with open(mano_calib_path, "r") as f:
            betas = yaml.safe_load(f)["betas"]

        pose_y = pose_npz["pose_y"]
        pose_m = pose_npz["pose_m"]
        ycb_ids = meta_item["ycb_ids"]

        equal_flags = np.all(np.all(pose_y == pose_y[0], axis=2), axis=0)

        # all seqs in dex-ycb contain only one moving object
        if np.count_nonzero(equal_flags) != len(ycb_ids) - 1:
            print(f"there is not only one moving object in subject: {subject_id}, seq: {seq_id}")
            print("this should not happen, please check. the program will exit")
            return None

        target_index = int(np.argmax(np.logical_not(equal_flags)))
        target_ycb_id = ycb_ids[target_index]
        assert target_index == meta_item["ycb_grasp_ind"]

        target_pose = pose_y[:, target_index]
        target_tl = target_pose[:, 4:]

        static_flags = np.logical_or(
            target_tl > target_tl[0] + np.ones(3) * moving_threshold,
            target_tl < target_tl[0] - np.ones(3) * moving_threshold,
        )
        # minus 1 is necessary
        end_frame = int(np.argmax(np.any(static_flags, axis=1))) - 1
        seq_path = os.path.join(subject_id, seq_id)

        valid_mano_flags = np.all(pose_m == 0, axis=2)[:, 0]
        start_frame = int(np.argmax(np.logical_not(valid_mano_flags)))
        if start_frame > end_frame:
            print(f"there is a invalid sequence. subject: {subject_id}, seq: {seq_id}")
            print("this sequence will be skipped.")
            continue

        # valid_frames_range includes end_frame
        result = {
            "seq_path": seq_path,
            "subject_id": subject_id,
            "seq_id": seq_id,
            "target_index": target_index,
            "target_ycb_id": target_ycb_id,
            "betas": betas,
            "valid_frames_range": [start_frame, end_frame],
            "num_valid_frames": int(end_frame - start_frame + 1),
        }
        result.update(meta_item)

        results[seq_id] = result

    # sort by key
    results = {k: v for k, v in sorted(list(results.items()))}
    return results


def transform_pose_m(data_root, mano_dir, meta_item):
    seq_path = meta_item["seq_path"]

    data_dir = os.path.join(data_root, seq_path)
    mano_calib_path = os.path.join(data_root, "calibration", f"mano_{meta_item['mano_calib'][0]}", "mano.yml")

    pose_path = os.path.join(data_dir, "pose.npz")
    pose = np.load(pose_path, allow_pickle=True)
    pose_m, pose_y = pose["pose_m"], pose["pose_y"]

    valid_mano_flags = np.any(pose_m != 0, axis=2)[:, 0]
    target_index = meta_item["target_index"]

    mano_ts = ManoTransformation(side=meta_item["mano_sides"][0], mano_root=mano_dir)
    with open(mano_calib_path, "r") as f:
        mano_calib = yaml.safe_load(f)
    betas = np.array(mano_calib["betas"], dtype=np.float32)

    new_pose_m = pose_m.copy()
    for frame_idx in range(meta_item["num_frames"]):
        if not valid_mano_flags[frame_idx]:
            continue

        pose_y_item = pose_y[frame_idx][target_index]
        pose_m_item = pose_m[frame_idx][0].copy()
        q, t = pose_y_item[:4], pose_y_item[4:]
        rot_object = Rot.from_quat(q)
        inv_rot_object = rot_object.inv()
        inv_tl_object = rot_object.inv().apply(-t)

        new_pose_m_item = mano_ts.transform_pose(pose_m_item, betas, quat=inv_rot_object.as_quat(), tl=inv_tl_object)
        new_pose_m[frame_idx, 0] = new_pose_m_item

    return new_pose_m


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def augment_pose_m(data_root, mano_dir, pose_m_object_coord):
    pose_m_aug = {}
    for seq_idx, seq_id in enumerate(tqdm(meta)):
        pose_m = pose_m_object_coord[seq_id]

        meta_item = meta[seq_id]

        mano_calib_path = os.path.join(data_root, "calibration", f"mano_{meta_item['mano_calib'][0]}", "mano.yml")

        mano_ts = ManoTransformation(side=meta_item["mano_sides"][0], mano_root=mano_dir)
        with open(mano_calib_path, "r") as f:
            mano_calib = yaml.safe_load(f)
        betas = np.array(mano_calib["betas"], dtype=np.float32)

        valid_mano_flags = np.any(pose_m != 0, axis=2)[:, 0]

        all_new_pose_m = [pose_m]
        all_aug_rot_mat = [np.eye(3)]
        for aug_idx in range(10):
            random_rot_mat = rand_rotation_matrix()
            all_aug_rot_mat.append(random_rot_mat)
            random_rot = Rot.from_matrix(random_rot_mat)

            new_pose_m = pose_m.copy()
            for frame_idx in range(meta_item["num_frames"]):
                if not valid_mano_flags[frame_idx]:
                    continue

                pose_m_item = pose_m[frame_idx][0].copy()

                new_pose_m_item = mano_ts.transform_pose(pose_m_item, betas, quat=random_rot.as_quat(), tl=np.zeros(3))
                new_pose_m[frame_idx, 0] = new_pose_m_item

            all_new_pose_m.append(new_pose_m)
        all_new_pose_m = np.stack(all_new_pose_m)
        all_aug_rot_mat = np.stack(all_aug_rot_mat)
        pose_m_aug[f"{seq_id}_mano"] = all_new_pose_m.copy()
        pose_m_aug[f"{seq_id}_aug_rot"] = all_aug_rot_mat.copy()
    return pose_m_aug


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/jianglong/data/raw/dex-ycb")
    parser.add_argument(
        "--mano_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "mano_v1_2_models")
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    )
    args = parser.parse_args()
    assert os.path.exists(args.data_root), f"DexYCB data root {args.data_root} does not exist"
    os.makedirs(args.output_dir, exist_ok=True)

    print("Extracting meta information from raw data...")
    meta = extract_meta(args.data_root)
    print(f"Saving meta information to {os.path.normpath(os.path.join(args.output_dir, 'meta.json'))}")
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    pose_m_object_coord = {}
    print("Transforming MANO pose to object coordinate...")
    for subject_id in tqdm(meta):
        new_pose_m = transform_pose_m(args.data_root, args.mano_dir, meta[subject_id])
        pose_m_object_coord[subject_id] = new_pose_m
    output_path = os.path.join(args.output_dir, "pose_m_object_coord.npz")
    print(f"Saving transformed MANO pose to {os.path.normpath(output_path)}")
    np.savez_compressed(output_path, **pose_m_object_coord)

    print("Augmenting MANO pose...")
    pose_m_aug = augment_pose_m(args.data_root, args.mano_dir, pose_m_object_coord)
    output_path = os.path.join(args.output_dir, "pose_m_aug.npz")
    print(f"Saving augmented MANO pose to {os.path.normpath(output_path)}")
    np.savez_compressed(output_path, **pose_m_aug)

    print("Copy calibration files...")
    shutil.copytree(os.path.join(args.data_root, "calibration"), os.path.join(args.output_dir, "calibration"))
    print("Copy model files (only textured_simple.obj and points.xyz)")
    for model_name in os.listdir(os.path.join(args.data_root, "models")):
        src_dir = os.path.join(args.data_root, "models", model_name)
        tgt_dir = os.path.join(args.output_dir, "models", model_name)
        os.makedirs(tgt_dir, exist_ok=True)
        shutil.copy(os.path.join(src_dir, "textured_simple.obj"), os.path.join(tgt_dir, "textured_simple.obj"))
        shutil.copy(os.path.join(src_dir, "points.xyz"), os.path.join(tgt_dir, "points.xyz"))
    print("Done")
