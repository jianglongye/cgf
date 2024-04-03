import copy
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Union

import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.structures import Meshes
from torch import nn

try:
    from pytorch3d.ops import knn_points
except ImportError:
    print("robotics.py: Unable to import knn_points")

from cgf.transformation import axis_angle_to_matrix


class KinematicsLayer(nn.Module):
    def __init__(
        self,
        urdf_str_or_path: str,
        end_links: List[str],
        global_transform: bool = False,
        geometry_path: str = None,
        geometry_data=None,
        return_geometry: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        if os.path.exists(urdf_str_or_path):
            with open(urdf_str_or_path) as f:
                chain: pk.chain.Chain = pk.build_chain_from_urdf(f.read())
        else:
            chain: pk.chain.Chain = pk.build_chain_from_urdf(urdf_str_or_path)

        if return_geometry and (geometry_path is None and geometry_data is None):
            raise TypeError("geometry_path or geometry_data should be set if geometry need to be returned")

        self.geometries: Dict[str, Meshes] = {}
        if return_geometry:
            # WARNING! we assume the geometry file contains all links
            if geometry_path is not None:
                geometry_data = np.load(geometry_path)
                geo_link_names = [name.replace("_verts", "") for name in geometry_data.files if "verts" in name]
            else:
                geo_link_names = [name.replace("_verts", "") for name in geometry_data if "verts" in name]
            for name in geo_link_names:
                verts, faces = geometry_data[name + "_verts"], geometry_data[name + "_faces"]
                verts, faces = torch.from_numpy(verts).to(dtype), torch.from_numpy(faces).long()
                mesh = Meshes(verts=[verts], faces=[faces]).to(device)
                self.geometries[name] = mesh

        self.end_links = end_links
        self.serial_chains: List[pk.chain.SerialChain] = []
        self.return_geometry = return_geometry
        self.global_transform = global_transform

        for link_name in end_links:
            serial_chain = pk.SerialChain(chain, link_name)
            serial_chain = serial_chain.to(dtype=dtype, device=device)
            self.serial_chains.append(serial_chain)

        self.dof: int = len(chain.get_joint_parameter_names())

    def forward(self, qpos: torch.Tensor):
        tf3ds: Dict[str, pk.Transform3d] = {}
        device, dtype, batch_size = qpos.device, qpos.dtype, qpos.shape[0]

        identiy = torch.eye(4, device=device, dtype=dtype)
        tf3ds["base_link"] = pk.Transform3d(batch_size, matrix=identiy)

        identiy = torch.eye(4, device=device, dtype=dtype)
        identiy[:3, 3] = torch.tensor([0, 0, 0], dtype=dtype, device=device)
        tf3ds["palm"] = pk.Transform3d(batch_size, matrix=identiy)

        start = 0 if not self.global_transform else 6
        for _, serial_chain in enumerate(self.serial_chains):
            # hard code for now
            joint_num = 4
            tf3ds.update(serial_chain.forward_kinematics(qpos[:, start : start + joint_num], end_only=False))
            start += joint_num

        if self.global_transform:
            rot_mat = axis_angle_to_matrix(qpos[:, 3:6])
            glb_tf3d = pk.Transform3d(rot=rot_mat, pos=qpos[:, 0:3], device=device, dtype=dtype)
            tf3ds = {name: glb_tf3d.compose(tf3d) for name, tf3d in tf3ds.items()}

        if not self.return_geometry:
            return tf3ds
        else:
            geos: Dict[str, Meshes] = {}
            for name in self.geometries:
                geo = self.geometries[name].extend(batch_size).clone()
                geo = geo.update_padded(tf3ds[name].transform_points(geo.verts_padded()))
                # geo = geo.update_padded(cvee.rotate(tf3ds[name].transform_points(geo.verts_padded()), debug_mat[None]))
                geos[name] = geo
            return tf3ds, geos


def load_joint_limits(urdf_path: str) -> Dict[str, List[float]]:
    with open(urdf_path) as f:
        urdf = ET.parse(f)
    joints = urdf.findall("joint")
    limits = {}
    for joint in joints:
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        lower_bound = float(limit.get("lower"))
        upper_bound = float(limit.get("upper"))
        limits[joint.get("name")] = [lower_bound, upper_bound]
    return limits


def pt3d_knn(src_xyz, trg_xyz, k=1):
    """
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, 3000] tensor for NN distance and index in N2
    """
    B = src_xyz.size(0)

    # [B], N for each num
    src_lengths = torch.full((src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device)
    trg_lengths = torch.full((trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device)

    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]

    return nn_dists, nn_idx


def batched_index_select(input, index, dim=1):
    """
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    """
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)


def get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
    """
    :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    """
    N1, N2 = src_xyz.size(1), trg_xyz.size(1)

    # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
    NN_src_xyz = batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
    NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
    NN_src_normal = batched_index_select(src_face_normal, trg_NN_idx)

    # interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false

    interior = F.cosine_similarity(NN_vector, NN_src_normal, dim=-1) > 0.2
    return interior


def self_intersection_loss(geos: Dict[str, Meshes], groups: List[List[str]], other_groups: List[List[str]]):
    if len(groups) != len(other_groups):
        raise RuntimeError("the length between groups and other groups should be the same")

    group_verts = [torch.cat([geos[n].verts_padded() for n in g], dim=1) for g in groups]
    group_normals = [torch.cat([geos[n].verts_normals_padded() for n in g], dim=1) for g in groups]
    other_groups_verts = [torch.cat([geos[n].verts_padded() for n in g], dim=1) for g in other_groups]

    batch_size = group_verts[0].shape[0]
    loss = 0
    for group_idx in range(len(group_verts)):
        nn_dist, nn_idx = pt3d_knn(other_groups_verts[group_idx], group_verts[group_idx])

        interior = get_interior(group_normals[group_idx], group_verts[group_idx], other_groups_verts[group_idx], nn_idx)
        interior = interior.to(torch.bool)

        penetr_dist = nn_dist[interior].sum()
        loss += penetr_dist

    loss /= batch_size
    return loss


def object_intersection_loss(joints, obj_points, obj_normals):
    batch_size = joints.shape[0]
    nn_dist, nn_idx = pt3d_knn(joints, obj_points)
    interior = get_interior(obj_normals, obj_points, joints, nn_idx)
    interior = interior.to(torch.bool)
    penetr_dist = nn_dist[interior].sum()
    loss = penetr_dist / batch_size
    return loss


def object_contact_loss(joints, obj_points):
    batch_size = joints.shape[0]
    nn_dist, _ = pt3d_knn(joints, obj_points)
    dist = nn_dist.sum()
    loss = dist / batch_size
    return loss


def rescale_urdf(urdf, scale):
    urdf = copy.deepcopy(urdf)
    links = urdf.findall("link")
    joints = urdf.findall("joint")

    for link in links:
        origin = link.find("collision/origin")
        if origin is not None:
            origin.set("xyz", " ".join([str(float(x) * scale) for x in origin.get("xyz").split()]))
        box = link.find("collision/geometry/box")
        if box is not None:
            box.set("size", " ".join([str(float(x) * scale) for x in box.get("size").split()]))
        sphere = link.find("collision/geometry/sphere")
        if sphere is not None:
            sphere.set("radius", str(float(sphere.get("radius")) * scale))

    for joint in joints:
        origin = joint.find("origin")
        if origin is not None:
            origin.set("xyz", " ".join([str(float(x) * scale) for x in origin.get("xyz").split()]))
    return urdf


def generate_geometry(urdf, remash=True, verbose=False):
    links = urdf.findall("link")
    geos = {}
    for link in links:
        origin = link.find("collision/origin")
        box = link.find("collision/geometry/box")
        sphere = link.find("collision/geometry/sphere")
        rpy = None if origin is None else [float(x) for x in origin.get("rpy").split()]
        xyz = None if origin is None else [float(x) for x in origin.get("xyz").split()]
        size = None if box is None else [float(x) for x in box.get("size").split()]
        radius = None if sphere is None else float(sphere.get("radius"))
        assert not (box is None and sphere is None)
        type = "box" if box is not None else "sphere"
        name = link.get("name")
        geos[name] = {"rpy": rpy, "xyz": xyz, "size": size, "radius": radius, "type": type}

    result = {}
    for link_name, geo in geos.items():
        if geo["type"] == "box":
            box = trimesh.creation.box(geo["size"])
            vert = np.asarray(box.vertices)
            faces = np.asarray(box.faces)
            if remash:
                if "base" in link_name:
                    vert, faces = trimesh.remesh.subdivide_to_size(vert, faces, 0.015)
                else:
                    vert, faces = trimesh.remesh.subdivide_to_size(vert, faces, 0.0075)
            if verbose:
                print(f"{link_name} box: {vert.shape}")
            xyz = geo["xyz"]
            vert += np.array(xyz)
            assert np.all(np.array(geo["rpy"]) == 0)

            result[f"{link_name}_verts"] = vert
            result[f"{link_name}_faces"] = faces
        else:
            if remash:
                sphere = trimesh.creation.icosphere(3, geo["radius"])
            else:
                sphere = trimesh.creation.icosphere(1, geo["radius"])
            vert = np.asarray(sphere.vertices)
            faces = np.asarray(sphere.faces)
            if verbose:
                print(f"{link_name} sphere: {vert.shape}")

            if geo["xyz"] is not None:
                assert np.all(np.array(geo["xyz"]) == 0)
            if geo["rpy"] is not None:
                assert np.all(np.array(geo["rpy"]) == 0)
            result[f"{link_name}_verts"] = vert
            result[f"{link_name}_faces"] = faces

    return result


class LPFilter:
    def __init__(self, control_freq, cutoff_freq):
        dt = 1 / control_freq
        wc = cutoff_freq * 2 * np.pi
        y_cos = 1 - np.cos(wc * dt)
        self.alpha = -y_cos + np.sqrt(y_cos**2 + 2 * y_cos)
        print(self.alpha)
        self.y = 0
        self.is_init = False

    def next(self, x):
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def init(self, y):
        self.y = y.copy()
        self.is_init = True
