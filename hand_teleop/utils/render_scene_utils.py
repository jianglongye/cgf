import os
from typing import List, Optional, Union

import cv2
import numpy as np
import open3d as o3d
from sapien import core as sapien
from sapien.core.pysapien import renderer as R

from hand_teleop.utils.mesh_utils import compute_smooth_shading_normal_np

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector


def segment_articulation_from_mask(segmentation: np.ndarray, articulation: sapien.ArticulationBase):
    link_id = [link.get_id() for link in articulation.get_links()]
    min_id = min(link_id)
    max_id = max(link_id)
    articulation_mask = np.logical_and(segmentation >= min_id, segmentation <= max_id)
    return articulation_mask


def render_geometry_to_open3d_mesh(render_shape: sapien.RenderShape, is_collision_mesh, scale):
    mesh = render_shape.mesh
    material = render_shape.material

    # TODO: consider other texture type other than diffuse
    has_material = len(material.diffuse_texture_filename) > 0 and not is_collision_mesh

    vertices = mesh.vertices
    indices = np.reshape(mesh.indices, [-1, 3]).astype(np.int32)
    normals = mesh.normals

    triangle_mesh = o3d.geometry.TriangleMesh(Vector3dVector(vertices * scale[None, :]), Vector3iVector(indices))
    triangle_mesh.vertex_normals = Vector3dVector(normals)
    if is_collision_mesh:
        if has_material:
            img = cv2.imread(material.diffuse_texture_filename)
            triangle_mesh.textures = o3d.geometry.Image(img)
        else:
            vertex_color = material.base_color[:3]
            triangle_mesh.vertex_colors = Vector3dVector(np.tile(vertex_color, (vertices.shape[0], 1)))
    return triangle_mesh


def merge_o3d_meshes(meshes: List[o3d.geometry.TriangleMesh]):
    vertices = []
    indices = []
    normals = []
    vertex_colors = []
    index = 0
    for mesh in meshes:
        vertices.append(np.asarray(mesh.vertices))
        indices.append(np.asarray(mesh.triangles) + index)
        normals.append(np.asarray(mesh.vertex_normals))
        vertex_colors.append(np.asarray(mesh.vertex_colors))
        index += vertices[-1].shape[0]
    vertices = np.concatenate(vertices)
    indices = np.concatenate(indices)
    normals = np.concatenate(normals)
    vertex_colors = np.concatenate(vertex_colors)
    triangle_mesh = o3d.geometry.TriangleMesh(Vector3dVector(vertices), Vector3iVector(indices))
    triangle_mesh.vertex_normals = Vector3dVector(normals)
    triangle_mesh.vertex_colors = Vector3dVector(vertex_colors)
    return triangle_mesh


def actor_to_open3d_mesh(actor: sapien.ActorBase, use_collision_mesh=False):
    meshes = []
    if not use_collision_mesh:
        for render_body in actor.get_visual_bodies():
            body_pose = render_body.local_pose.to_transformation_matrix()
            render_body_scale = render_body.scale
            for render_shape in render_body.get_render_shapes():
                triangle_mesh = render_geometry_to_open3d_mesh(
                    render_shape, is_collision_mesh=use_collision_mesh, scale=render_body_scale
                )
                triangle_mesh.transform(body_pose)
                meshes.append(triangle_mesh)
    else:
        for collision_visual_body in actor.get_collision_visual_bodies():
            body_pose = collision_visual_body.local_pose.to_transformation_matrix()
            for render_shape in collision_visual_body.get_render_shapes():
                triangle_mesh = render_geometry_to_open3d_mesh(render_shape)
                triangle_mesh.transform(body_pose)
                meshes.append(triangle_mesh)

    if len(meshes) > 0:
        mesh = merge_o3d_meshes(meshes)
        mesh.transform(actor.get_pose().to_transformation_matrix())
    else:
        mesh = None
    return mesh


def duplicate_actor_as_vulkan_nodes(actor: sapien.ActorBase, scene: sapien.Scene, use_shadow=True, opacity=None):
    render_scene: R.Scene = scene.get_renderer_scene()._internal_scene
    parent_node = render_scene.add_node(parent=None)
    parent_node.set_position(actor.get_pose().p)
    parent_node.set_rotation(actor.get_pose().q)
    nodes = [parent_node]
    for render_body in actor.get_visual_bodies():
        body_pose = render_body.local_pose
        vulkan_objects = render_body._internal_objects
        for vulkan_object in vulkan_objects:
            vulkan_model = vulkan_object.model
            node = render_scene.add_object(vulkan_model, parent_node)
            node.set_position(body_pose.p)
            node.set_rotation(body_pose.q)
            node.set_scale(vulkan_object.scale)
            node.transparency = 1 - opacity
            if use_shadow:
                node.shading_mode = 0
                node.cast_shadow = True
            nodes.append(node)
    return nodes


def export_scene_as_multiple_meshes(scene: sapien.Scene, directory: str, init_num=0):
    num = init_num
    os.makedirs(directory, exist_ok=True)
    for actor in scene.get_all_actors():
        if any([name in actor.get_name() for name in ["mug", "ground"]]):
            continue
        else:
            mesh = actor_to_open3d_mesh(actor)
            if mesh is not None:
                o3d.io.write_triangle_mesh(f"{directory}/{num:0>3d}.obj", mesh)
                num += 1
    for articulation in scene.get_all_articulations():
        meshes = []
        for actor in articulation.get_links():
            mesh = actor_to_open3d_mesh(actor)
            if mesh is not None:
                meshes.append(mesh)
        art_mesh = merge_o3d_meshes(meshes)
        o3d.io.write_triangle_mesh(f"{directory}/{num:0>3d}.obj", art_mesh, write_ascii=True)
        num += 1
    return num


def set_entity_visibility(entities: List[Union[sapien.ActorBase, sapien.ArticulationBase]], visibility: float):
    for entity in entities:
        if isinstance(entity, sapien.ActorBase):
            for geom in entity.get_visual_bodies():
                geom.set_visibility(visibility)
        elif isinstance(entity, sapien.ArticulationBase):
            for actor in entity.get_links():
                for geom in actor.get_visual_bodies():
                    geom.set_visibility(visibility)
        else:
            raise ValueError(f"Unrecognized type {type(entity)}")


def set_entity_color(entities: List[Union[sapien.ActorBase, sapien.ArticulationBase]], color: List[float]):
    if len(color) != 4:
        raise ValueError(f"RGBA Color should be a length 4 iterable")
    for entity in entities:
        if isinstance(entity, sapien.ActorBase):
            for geom in entity.get_visual_bodies():
                for shape in geom.get_render_shapes():
                    mat = shape.material
                    mat.set_base_color(np.array(color))
                    shape.set_material(mat)
        elif isinstance(entity, sapien.ArticulationBase):
            for actor in entity.get_links():
                for geom in actor.get_visual_bodies():
                    for shape in geom.get_render_shapes():
                        mat = shape.material
                        mat.set_base_color(np.array(color))
                        shape.set_material(mat)
        else:
            raise ValueError(f"Unrecognized type {type(entity)}")


def add_mesh_to_renderer(
    scene: sapien.Scene,
    renderer: sapien.VulkanRenderer,
    vertex: np.ndarray,
    faces: np.ndarray,
    material: R.Material,
    parent: Optional[R.Node] = None,
):
    context: R.Context = renderer._internal_context
    render_scene: R.Scene = scene.get_renderer_scene()._internal_scene
    normals = compute_smooth_shading_normal_np(vertex, faces)
    mesh = context.create_mesh_from_array(vertex, faces, normals)
    model = context.create_model([mesh], [material])
    if parent is not None:
        obj = render_scene.add_object(model, parent)
    else:
        obj = render_scene.add_object(model)
    return obj


def add_line_set_to_renderer(
    scene: sapien.Scene,
    renderer: sapien.VulkanRenderer,
    position: np.ndarray,
    connection: np.ndarray,
    color: np.ndarray = np.ones(4),
    parent: Optional[R.Node] = None,
):
    num_point = position.shape[0]
    if connection.shape[1] != 2:
        raise ValueError(f"Connection should be a mx2 array, but now get {connection.shape}")
    if position.shape[1] != 3:
        raise ValueError(f"Connection should be a nx3 array, but now get {position.shape}")
    if np.max(connection) > num_point:
        raise IndexError(f"Index in connection exceed the number of position")
    context: R.Context = renderer._internal_context
    render_scene: R.Scene = scene.get_renderer_scene()._internal_scene
    edge = position[connection.reshape([-1])]
    line_set = context.create_line_set(edge, np.tile(color, connection.size))
    if parent is not None:
        obj = render_scene.add_line_set(line_set, parent)
    else:
        obj = render_scene.add_line_set(line_set)
    return obj


def test_mesh_function():
    engine = sapien.Engine()
    renderer = sapien.VulkanRenderer(offscreen_only=True)
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    builder = scene.create_actor_builder()
    builder.add_box_visual(pose=sapien.Pose([0, 0, 1.5]), half_size=[0.5, 0.5, 0.5])
    builder.add_capsule_visual(radius=0.1, half_length=0.5)
    builder.add_sphere_visual(radius=1)
    builder.add_sphere_collision()
    actor = builder.build()

    mesh = actor_to_open3d_mesh(actor)
    o3d.io.write_triangle_mesh("hhh.obj", mesh)


if __name__ == "__main__":
    test_mesh_function()
