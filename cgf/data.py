import json
import os.path

import numpy as np
import torch
import trimesh
import yaml
from torch.utils.data import Dataset

YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
    # 22: "100_ball",
}

YCB_SIZE = {
    "002_master_chef_can": (0.1025, 0.1023, 0.1401),
    "003_cracker_box": (0.2134, 0.1640, 0.0717),
    "004_sugar_box": (0.0495, 0.0940, 0.1760),
    "005_tomato_soup_can": (0.0677, 0.0679, 0.1018),
    "006_mustard_bottle": (0.0576, 0.0959, 0.1913),
    "010_potted_meat_can": (0.0576, 0.1015, 0.0835),
    "011_banana": (0.1088, 0.1784, 0.0366),
    "021_bleach_cleanser": (0.1024, 0.0677, 0.2506),
    "024_bowl": (0.1614, 0.1611, 0.0550),
    "025_mug": (0.1169, 0.0930, 0.0813),
    "051_large_clamp": (0.1659, 0.1216, 0.0364),
    "035_power_drill": (0.1873, 0.1842, 0.0573),
    "019_pitcher_base": (0.1448, 0.1490, 0.2426),
    "061_foam_brick": (0.0778, 0.0526, 0.0511),
    # "100_ball": (0.1000, 0.1000, 0.1000),
}


def rand_indices(size, sample_num, is_sorted=True):
    if size >= sample_num:
        rand_indices = np.random.choice(size, sample_num, replace=False)
    else:
        rand_indices = np.random.choice(size, sample_num, replace=True)
    if is_sorted:
        rand_indices = sorted(rand_indices)
    return rand_indices


def sample(array, sample_num, dim=0, is_sorted=True):
    return_tensor = isinstance(array, torch.Tensor)

    array_num = array.shape[dim]
    if not return_tensor:
        if array_num >= sample_num:
            rand_indices = np.random.choice(array_num, sample_num, replace=False)
        else:
            rand_indices = np.random.choice(array_num, sample_num, replace=True)
        if is_sorted:
            rand_indices = sorted(rand_indices)
        return np.take(array, rand_indices, axis=dim)


class AllegroDataset(Dataset):
    def __init__(
        self,
        processed_data_root,
        target_ycb_ids="all",
        target_mano_side="right",
        augment_data=True,
        last_frame_only=False,
        min_seq_num=10,
    ):
        super().__init__()

        assert target_mano_side == "left", "change retargeting_dir manually if change the mano side"

        if target_ycb_ids == "all":
            target_ycb_ids = list(YCB_CLASSES.keys())
        elif target_ycb_ids == "train":
            target_ycb_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21]
        elif target_ycb_ids == "test":
            target_ycb_ids = [10, 11, 12, 13, 14]
        else:
            target_ycb_ids = [target_ycb_ids]

        self.last_frame_only = last_frame_only

        meta_path = os.path.join(processed_data_root, "meta.json")
        pose_m_aug_path = os.path.join(processed_data_root, "pose_m_aug.npz")
        retargeting_dir = os.path.join(processed_data_root, f"retargeting_{target_mano_side}")
        ycb_model_dir = os.path.join(processed_data_root, "models")

        with open(meta_path, "r") as f:
            meta = json.load(f)
        pose_m_aug_npz_file = np.load(pose_m_aug_path)

        seq_ids = [x[:-5] for x in pose_m_aug_npz_file.files if "mano" in x]
        self.seqs_meta = {
            k: meta[k]
            for k in seq_ids
            if meta[k]["target_ycb_id"] in target_ycb_ids and meta[k]["mano_sides"][0] == target_mano_side
        }
        if target_ycb_ids != "all":
            sorted_seq_ids = sorted([x for x in seq_ids if meta[x]["mano_sides"][0] == target_mano_side])
            indices = [sorted_seq_ids.index(k) for k in self.seqs_meta]
            indices = torch.tensor(indices)

        self.seqs_meta = [v for k, v in sorted(self.seqs_meta.items(), key=lambda x: x[0])]

        self.per_seq_aug_num = 11 if augment_data else 1

        retargeting_data = [np.load(os.path.join(retargeting_dir, f"{i}.npz")) for i in range(self.per_seq_aug_num)]
        # (seq_num, aug_num, frame_num, qpos_dim)
        self.qpos = torch.from_numpy(np.stack([d["qpos"] for d in retargeting_data], axis=1)).float()
        # (seq_num, aug_num, 3, 3)
        self.aug_rot = torch.from_numpy(np.stack([d["aug_rot"] for d in retargeting_data], axis=1)).float()

        if target_ycb_ids != "all":
            assert len(sorted_seq_ids) == len(self.qpos), "retargeting data is not complete"
            self.qpos = self.qpos[indices]
            self.aug_rot = self.aug_rot[indices]

        valid_frame_mask = np.array([v["num_valid_frames"] > min_seq_num for v in self.seqs_meta])
        self.seqs_meta = [v for idx, v in enumerate(self.seqs_meta) if valid_frame_mask[idx]]
        valid_frame_mask = torch.from_numpy(valid_frame_mask).bool()
        self.qpos = self.qpos[valid_frame_mask]
        self.aug_rot = self.aug_rot[valid_frame_mask]

        assert self.qpos.shape[0] == len(self.seqs_meta)
        assert self.qpos.shape[0] == len(self.aug_rot)
        assert self.qpos.shape[1] == self.per_seq_aug_num
        print(f"valid seq num: {self.qpos.shape[0]}")
        print(f"aug num: {self.qpos.shape[1]}")
        print(f"max frame num: {self.qpos.shape[2]}")

        if self.last_frame_only:
            raise NotImplementedError
        else:
            self.seq_aug_tuples = [
                (seq_idx, aug_idx) for aug_idx in range(self.qpos.shape[1]) for seq_idx in range(self.qpos.shape[0])
            ]

        self.rand_index = []
        for seq_idx in range(self.qpos.shape[0]):
            num_valid_frames = self.seqs_meta[seq_idx]["num_valid_frames"]
            last_5_frames = np.arange(num_valid_frames - 5, num_valid_frames)
            other_frames = rand_indices(num_valid_frames, 20, is_sorted=True)
            self.rand_index.append(np.concatenate([last_5_frames, other_frames]))
            if num_valid_frames < self.qpos.shape[2]:
                assert torch.allclose(
                    self.qpos[seq_idx, :, num_valid_frames], torch.zeros_like(self.qpos[seq_idx, :, num_valid_frames])
                )
        self.rand_index = torch.from_numpy(np.array(self.rand_index)).long()

        qpos = [self.qpos[seq_idx, :, self.rand_index[seq_idx]] for seq_idx in range(self.rand_index.shape[0])]
        self.qpos = torch.stack(qpos, dim=0)

        self.obj_pts = {}
        for ycb_id in target_ycb_ids:
            ycb_name = YCB_CLASSES[ycb_id]
            point_path = os.path.join(ycb_model_dir, ycb_name, "points.xyz")
            pts = np.asarray(trimesh.load(point_path).vertices)
            rand_idx = np.random.choice(pts.shape[0], 2000, replace=False)
            self.obj_pts[ycb_id] = torch.from_numpy(pts[rand_idx]).float()

    def __len__(self):
        if self.last_frame_only:
            return len(self.seq_frame_aug_tuples)
        else:
            return len(self.seq_aug_tuples)

    def __getitem__(self, idx):
        if self.last_frame_only:
            raise NotImplementedError
        else:
            seq_idx, aug_idx = self.seq_aug_tuples[idx]
            meta_item = self.seqs_meta[seq_idx]
            seq_id = meta_item["seq_id"]
            ycb_id = meta_item["target_ycb_id"]

            num_valid_frames = meta_item["num_valid_frames"]
            qpos = self.qpos[seq_idx, aug_idx]
            rand_idx = self.rand_index[seq_idx]
            query_t = (num_valid_frames - 1 - rand_idx) / (num_valid_frames - 1)

            aug_rot = self.aug_rot[seq_idx, aug_idx]
            betas = torch.tensor(meta_item["betas"]).expand(qpos.shape[0], -1)
            obj_pts = self.obj_pts[ycb_id]

            return {
                "seq_id": seq_id,
                "aug_idx": aug_idx,
                "query_t": query_t,
                "qpos": qpos,
                "aug_rot": aug_rot,
                "obj_pts": obj_pts,
                "betas": betas,
            }


class DexYCBDataset(Dataset):
    def __init__(
        self,
        processed_data_root,
        target_ycb_ids=1,
        target_mano_side="left",
        augment_data=True,
        last_frame_only=False,
        return_seq=False,
        seq_num=20,
        min_seq_num=10,
    ):
        super(DexYCBDataset, self).__init__()
        assert target_mano_side in ["left", "right"]
        assert not (last_frame_only and return_seq)

        if target_ycb_ids == "all":
            target_ycb_ids = list(YCB_CLASSES.keys())
        else:
            target_ycb_ids = [target_ycb_ids]

        meta_path = os.path.join(processed_data_root, "meta.json")
        pose_m_aug_path = os.path.join(processed_data_root, "pose_m_aug.npz")
        calib_dir = os.path.join(processed_data_root, "calibration")
        ycb_model_dir = os.path.join(processed_data_root, "models")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        if augment_data:
            self.per_seq_aug_num = 11
        else:
            self.per_seq_aug_num = 1
        self.last_frame_only = last_frame_only
        self.seq_num = seq_num
        self.return_seq = return_seq

        # id: 1 - 21, around 50 sequences for one object
        self.seqs_meta = {
            k: v
            for k, v in meta.items()
            if v["target_ycb_id"] in target_ycb_ids and v["mano_sides"][0] == target_mano_side
        }

        # sequences statistics
        if self.return_seq:
            self.seqs_meta = {k: v for k, v in self.seqs_meta.items() if v["num_valid_frames"] >= min_seq_num}

            nums_valid_frames = [self.seqs_meta[k]["num_valid_frames"] for k in self.seqs_meta]
            print(f"valid seq num: {len(self.seqs_meta)}")
            print(
                f"frame num, mean: {sum(nums_valid_frames) / len(nums_valid_frames):.2f}, "
                f"min: {min(nums_valid_frames)}, max: {max(nums_valid_frames)}"
            )
            self.seq_aug_tuples = []
            for seq_id in self.seqs_meta:
                self.seq_aug_tuples.extend([(seq_id, aug_idx) for aug_idx in range(self.per_seq_aug_num)])
        else:
            self.seq_frame_aug_tuples = []
            for seq_id in self.seqs_meta:
                start_frame, end_frame = self.seqs_meta[seq_id]["valid_frames_range"]
                if last_frame_only:
                    start_frame = end_frame
                # including end_frame
                self.seq_frame_aug_tuples.extend(
                    [
                        (seq_id, aug_idx, idx)
                        for idx in range(start_frame, end_frame + 1)
                        for aug_idx in range(self.per_seq_aug_num)
                    ]
                )

        self.mano_betas = {}
        for seq_id in self.seqs_meta:
            meta_item = self.seqs_meta[seq_id]
            mano_calib_path = os.path.join(calib_dir, f"mano_{meta_item['mano_calib'][0]}", "mano.yml")
            with open(mano_calib_path, "r") as f:
                mano_calib = yaml.safe_load(f)
            self.mano_betas[seq_id] = torch.tensor(mano_calib["betas"].copy(), dtype=torch.float32)

        pose_m_aug_npz_file = np.load(pose_m_aug_path)
        self.pose_m_aug = {}
        self.aug_rot = {}
        for seq_id in self.seqs_meta:
            if self.return_seq:
                start_frame, end_frame = self.seqs_meta[seq_id]["valid_frames_range"]
                pose_m_aug = pose_m_aug_npz_file[f"{seq_id}_mano"][:, start_frame : end_frame + 1]
                assert pose_m_aug.shape[1] == self.seqs_meta[seq_id]["num_valid_frames"]
                if self.seq_num > 0:
                    sampled_pose_m_aug = sample(pose_m_aug, seq_num, dim=1, is_sorted=True)
                    self.pose_m_aug[seq_id] = torch.from_numpy(sampled_pose_m_aug).float()
                else:
                    self.pose_m_aug[seq_id] = torch.from_numpy(pose_m_aug).float()
            else:
                self.pose_m_aug[seq_id] = torch.from_numpy(pose_m_aug_npz_file[f"{seq_id}_mano"]).float()
            self.aug_rot[seq_id] = torch.from_numpy(pose_m_aug_npz_file[f"{seq_id}_aug_rot"]).float()

        self.obj_pts = {}
        for ycb_id in target_ycb_ids:
            ycb_name = YCB_CLASSES[ycb_id]
            point_path = os.path.join(ycb_model_dir, ycb_name, "points.xyz")
            pts = np.asarray(trimesh.load(point_path).vertices)
            rand_idx = np.random.choice(pts.shape[0], 2000, replace=False)
            self.obj_pts[ycb_id] = torch.from_numpy(pts[rand_idx]).float()

    def __len__(self):
        if self.return_seq:
            return len(self.seq_aug_tuples)
        else:
            return len(self.seq_frame_aug_tuples)

    def __getitem__(self, item):
        if self.return_seq:
            seq_id, aug_idx = self.seq_aug_tuples[item]
            meta_item = self.seqs_meta[seq_id]
            ycb_id = meta_item["target_ycb_id"]

            pose_m = self.pose_m_aug[seq_id][aug_idx][:, 0]
            query_coords = pose_m[:, 48:51].clone()
            aug_rot = self.aug_rot[seq_id][aug_idx]
            betas = self.mano_betas[seq_id].unsqueeze(0).expand(pose_m.shape[0], -1, -1)
            obj_pts = self.obj_pts[ycb_id]

            return {
                "seq_id": seq_id,
                "aug_idx": aug_idx,
                "pose_m": pose_m,
                "aug_rot": aug_rot,
                "obj_pts": obj_pts,
                "betas": betas,
                "query_coords": query_coords,
            }
        else:
            seq_id, aug_idx, frame_idx = self.seq_frame_aug_tuples[item]

            meta_item = self.seqs_meta[seq_id]
            ycb_id = meta_item["target_ycb_id"]

            pose_m = self.pose_m_aug[seq_id][aug_idx, frame_idx][0]
            query_coords = pose_m[48:51].clone()
            aug_rot = self.aug_rot[seq_id][aug_idx]
            betas = self.mano_betas[seq_id]
            obj_pts = self.obj_pts[ycb_id].clone()

            return {
                "seq_id": seq_id,
                "aug_idx": aug_idx,
                "frame_idx": frame_idx,
                "pose_m": pose_m,
                "aug_rot": aug_rot,
                "obj_pts": obj_pts,
                "betas": betas,
                "query_coords": query_coords,
            }

    def query_seq_data(self, seq_id, aug_idx):
        start_frame, end_frame = self.seqs_meta[seq_id]["valid_frames_range"]
        if self.last_frame_only:
            start_frame = end_frame

        meta_item = self.seqs_meta[seq_id]
        ycb_id = meta_item["target_ycb_id"]

        pose_m = self.pose_m_aug[seq_id][aug_idx, start_frame : end_frame + 1, 0]
        query_coords = pose_m[..., 48:51].clone()

        aug_rot = self.aug_rot[seq_id][aug_idx].unsqueeze(0).expand(end_frame - start_frame + 1, -1, -1)
        betas = self.mano_betas[seq_id].unsqueeze(0).expand(end_frame - start_frame + 1, -1)
        obj_pts = self.obj_pts[ycb_id].clone().unsqueeze(0).expand(end_frame - start_frame + 1, -1, -1)

        return {
            "sqe_id": seq_id,
            "aug_idx": aug_idx,
            "pose_m": pose_m,
            "aug_rot": aug_rot,
            "obj_pts": obj_pts,
            "betas": betas,
            "query_coords": query_coords,
        }


def get_ycb_mesh(raw_data_root, ycb_id):
    ycb_model_dir = os.path.join(raw_data_root, "models")
    obj_path = os.path.join(ycb_model_dir, YCB_CLASSES[ycb_id], "textured_simple.obj")
    obj_mesh = trimesh.load(obj_path)
    return np.asarray(obj_mesh.vertices), np.asarray(obj_mesh.faces)
