import sys
import os
import gradio as gr
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import math

# replace this with the path to the MoGe repository
sys.path.append('./MoGe')
from moge.model import MoGeModel
from moge.utils.vis import colorize_depth
import utils3d

from glob import glob
from pathlib import Path

# these are my utility functions, they should get installed as part of the intrinsic repo
from chrislib.data_util import load_image, np_to_pil, load_from_url
from chrislib.general import show, invert, uninvert, to2np, match_scale, rescale

from altered_midas.midas_net import MidasNet

from intrinsic.pipeline import load_models, run_pipeline

from skimage.transform import resize
from PIL import Image

import numpy as np
import mitsuba as mi
import drjit as dr
import cv2
import torch
import json
import pickle
import re
import tempfile
import matplotlib.pyplot as plt
from dict2xml import dict2xml
from scipy.spatial.transform import Rotation as R

import trimesh
import open3d as o3d
from typing import Tuple
from utils import *

# make temp directory to save render results
temp_dir = tempfile.mkdtemp()


def intrinsic_decomposition(intrinsic_model, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose the input image into albedo and shading components.
    Args:
        intrinsic_model: The intrinsic decomposition model.
        img (np.ndarray): The input image.
    Returns:
        image (np.ndarray): The input image.
        albedo (np.ndarray): The albedo component.
        dif_shd (np.ndarray): The shading component.
    """

    intrinsic_result = run_pipeline(
        intrinsic_model,
        img,
        linear=False,
        resize_conf=1024
    )

    albedo = intrinsic_result['hr_alb']
    hr_shd = intrinsic_result['hr_shd']
    dif_shd = intrinsic_result['dif_shd']
    image = intrinsic_result['image']
    # show([image, albedo, dif_shd])
    return image, albedo, dif_shd


def geometry_reconstruction(moge_model: MoGeModel, image: np.ndarray, height: int, width: int, sub_h: int, sub_w: int, albedo: np.ndarray, device: str, threshold: float) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct geometry from the input image using the MoGe model.
    Args:
        moge_model (MoGeModel): The MoGe model.
        image (np.ndarray): The input image.
        height (int): Height of the input image.
        width (int): Width of the input image.
        sub_h (int): Height of the sub-image.
        sub_w (int): Width of the sub-image.
        albedo (np.ndarray): The albedo component.
        device (str): Device to run the model on ('cuda' or 'cpu').
        threshold (float): Threshold for cutting mesh edges.
    Returns:
        intrinsics (np.ndarray): Camera intrinsics.
        mask (np.ndarray): Mask of valid part of the image.
        sky_comp_mask (np.ndarray): Mask of sky part of the image.
        edge_mask (np.ndarray): Mask of edge part of the image.
    """
    # generate points, depth, mask, and intrinsics from MoGe model
    image_tensor = torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1)
    output = moge_model.infer(image_tensor ** (1/2.2))
    points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
    points = points.clip(max=2**32)
    # resize to original image size
    points = resize(points, (height, width))
    depth = resize(depth, (height, width))
    mask = resize(mask, (height, width))
    # compute sky mask
    sky_mask = ~mask
    sky_comp_mask = sky_mask[..., None].astype(np.single)
    sub_sky_msk = resize(sky_comp_mask, (sub_h, sub_w))
    # construct mesh from points and texture it with albedo
    faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
        points,
        albedo,
        utils3d.numpy.image_uv(width=width, height=height),
        mask=mask & ~utils3d.numpy.depth_edge(depth, rtol=threshold, mask=mask),
        tri=True
    )
    edge_mask = ~utils3d.numpy.depth_edge(depth, rtol=threshold, mask=mask)

    # (from MoGe repository)
    # when exporting the model, follow the OpenGL coordinate conventions:
    # - world coordinate system: x right, y up, z backward.
    # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
    vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

    # this is a silly way to get mitsuba to read the mesh properly
    save_ply('mesh.ply', vertices, faces, vertex_colors)
    mesh = o3d.io.read_triangle_mesh('mesh.ply')
    o3d.io.write_triangle_mesh('mesh.ply', mesh)

    return intrinsics, mask, sky_comp_mask, edge_mask


def prepare_diffren_scene(intrinsics: np.ndarray, sub_h: int, sub_w: int, height: int, width: int) -> dict:
    """
    Prepare the scene dictionary for Mitsuba rendering.
    Args:
        intrinsics (np.ndarray): Camera intrinsics.
        sub_h (int): Height of the sub-image.
        sub_w (int): Width of the sub-image.
        height (int): Height of the original image.
        width (int): Width of the original image.
    Returns:
        scene_dict (dict): Scene dictionary for Mitsuba rendering.
    """
    mi.set_variant('cuda_ad_rgb') # can be set to llvm_ad_rgb for CPU rendering

    # load the .ply file
    mesh_dict = {
        'type': 'ply',
        'filename': 'mesh.ply',
        'bsdf':
        {
            'type': 'diffuse',
            'reflectance':
            {
                'type': 'mesh_attribute',
                'name': 'vertex_color'
            }
        }
    }

    mesh = mi.load_dict(mesh_dict)
    mesh_params = mi.traverse(mesh)

    mesh_params['vertex_color'] = (mesh_params['vertex_color'] / 255.0)

    mesh_params.update()

    # use the MoGe code to convert camera intrinsic matrix to field of view
    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
    fov_x, fov_y = np.rad2deg(fov_x), np.rad2deg(fov_y)

    print(f"FOV X: {fov_x}, FOV Y: {fov_y}")

    cam_dict1 = {
        'type': 'perspective',
        'fov': float(fov_y),
        'fov_axis' : 'y',
        'to_world': mi.ScalarTransform4f().look_at(
            mi.ScalarPoint3f([0, 0, 0]),  # camera at origin
            mi.ScalarPoint3f([0, 0, -1]), # looking down the -z axis (?) not sure about mitsuba conventions
            mi.ScalarPoint3f([0, 1, 0])
        ),
        # we can render at a smaller scale even though our mesh is computed at the original image size
        'film': {
            'type': 'hdrfilm',
            'width': sub_w,
            'height': sub_h
        }
    }
    cam_dict2 = {
        'type': 'perspective',
        'fov': float(fov_y),
        'fov_axis' : 'y',
        'to_world': mi.ScalarTransform4f().look_at(
            mi.ScalarPoint3f([0, 0, 0]),  # camera at origin
            mi.ScalarPoint3f([0, 0, -1]), # looking down the -z axis (?) not sure about mitsuba conventions
            mi.ScalarPoint3f([0, 1, 0])
        ),
        # this camera will render the scene at the full original resolution
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height
        }
    }

    # we first create the scene with the "aov" integrator to render the position map
    # this way we have the mesh's 3D points in image space which makes it easier to place the lights
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': 3
        },
        'sensor1': cam_dict1,
        'sensor2': cam_dict2,
        'envmap': {
            'type': 'envmap',
            'bitmap' : mi.Bitmap(np.ones((128, 256, 3)) * 1),
        },
        'scene_mesh': mesh,
        'sampler': {
            'type': 'orthogonal'
        }
    }

    return scene_dict


def optimize_light(scene_dict: dict, target, lights_mask: np.ndarray) -> Tuple[dict, np.ndarray]:
    """
    Infer the lighting conditions of the scene using differentiable rendering.
    Args:
        scene_dict (dict): Scene dictionary for Mitsuba rendering.
        target (mi.TensorXf): Target image for optimization.
        lights_mask (np.ndarray): Mask of the light sources specified by users.
    Returns:
        light_dict (dict): Dictionary of optimized light sources configurations.
        opt_envmap_data (np.ndarray): Optimized environment map data.
    """
    scene_dict = scene_dict.copy()
    scene = mi.load_dict(scene_dict)

    sub_w, sub_h = scene.sensors()[0].film().crop_size()

    # Calculate connected regions of the input mask
    lights_mask = resize(lights_mask, [sub_h, sub_w])
    binary_mask = (lights_mask[:, :, 3] > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

    connected_areas = []
    pixel_areas = {}
    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        connected_areas.append({"label": i, "bbox": (x, y, w, h), "centroid": (cx, cy), "area": area})

        pixel_indices = np.argwhere(labels == i)  # Get pixel coordinates
        pixel_areas[i] = pixel_indices  # Store pixels for each label
    
    # Update scene dict with initial lights
    pl_count = 0
    if len(connected_areas) == 0: # if no estimated light source input
        print("Estimated Lighting Not Specified")
        grid_size = 4 # size of point light grid (grid_size x grid_size)
        margin = 50 # margin around the image for the grid

        aov_out = render_position_and_normal(scene_dict, sensor=0)

        positions = aov_out[:, :, :3]
        normals = aov_out[:, :, 3:6]
        # create a grid of evenly spaced values in image space (with some margin)
        im_grid = np.meshgrid(np.linspace(margin, sub_w-(margin), grid_size), np.linspace(margin, sub_h-(margin), grid_size))

        for i, (im_x, im_y) in enumerate(zip(im_grid[0].flatten(), im_grid[1].flatten())):

            # get the 3D position of the point in world space
            pos = np.array(positions[int(im_y) - 1, int(im_x) - 1])
            nrm = np.array(normals[int(im_y) - 1, int(im_x) - 1])
            # here I'm backing the lights a bit off the surface towards the camera
            # a better way would be to render out the normals in the AOV pass, then use those
            pos += nrm * 0.05
            print(pos)
            # create RGB point lights for each grid location, make them dim
            scene_dict[f'pointlight_{pl_count}'] = {
                'type': 'point',
                'position': list(pos),
                'intensity': {
                    'type': 'rgb',
                    'value': [0.01, 0.01, 0.01]
                }
            }
            pl_count += 1
    else: # users specified light source
        # get the positions of lights inserted
        space = 0.03
        for label, pixels in pixel_areas.items():
            # choole pixels
            num_samples = 10 
            step = max(1, len(pixels) // num_samples)  # Adjust step size based on density
            sampled_pixels = pixels[::step]  # Select every 'step' pixel
            # place point lights
            for i in range(num_samples):
                coord = sampled_pixels[i].astype(int).tolist()
                coord[0], coord[1] = coord[1], coord[0]
                position, normal = get_position_normal(scene_dict, coord, 0) # get location and normal for each pixel
                if position is None: continue
                scene_dict[f'pointlight_{pl_count}'] = {
                    'type': 'point',
                    'position': np.array(list(position + space*normal)) * np.array([-1, 1, -1]),
                    'intensity': {
                        'type': 'rgb',
                        'value': [0.08, 0.08, 0.08]
                    }
                }
                pl_count += 1

    # Optimize lights through differentiable rendering
    scene_dict['integrator'] = {
            'type': 'path',
            'max_depth': 3
        }
    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)

    keys = []
    for i in range(pl_count):
        keys.append(f'pointlight_{i}.position')
        keys.append(f'pointlight_{i}.intensity.value')
    keys.append('envmap.data')
    learning_rate = 0.01
    opt = mi.ad.Adam(lr=learning_rate, mask_updates=True, beta_1=0.8)
    for k in keys:
        dr.enable_grad(params[k])
        opt[k] = params[k]

    # optimization
    MAX_ITERATIONS = 150
    errors = []
    curr_loss = float('inf')
    best_loss = float('inf')
    patience = 0
    MAX_PATIENCE = 5

    curr_spp = 16

    for it in range(MAX_ITERATIONS):
        params.update(opt)

        # Perform a (noisy) differentiable rendering of the scene
        rendered = mi.render(scene, params, spp=curr_spp)

        # Evaluate the objective function from the current rendered image
        loss = mse(
            rendered,
            target,
        )

        # Backpropagate through the rendering process
        dr.backward(loss)
        opt.step()

        # ensure that light intensities stay positive
        for k in keys:
            if 'position' not in k:
                opt[k] = dr.maximum(opt[k], 0.001)

        if loss < best_loss:
            best_loss = loss
            patience = 0
        else:
            patience += 1
            learning_rate *= 0.5
            opt.set_learning_rate(learning_rate)

        if patience >= MAX_PATIENCE:
            break

        print(f"Iteration {it:02d}: {loss}", end='\r')

    light_dict = {}
    # save the optimized parameters
    for key in scene_dict.keys():
        if key.find("pointlight") != -1:
            light_dict[key] = {
                'type': 'point',
                'position': list(np.asarray(params[key + '.position']).flatten()),
                'intensity': {
                    'type': 'rgb',
                    'value': list(np.asarray(params[key + '.intensity.value']).flatten())
                }
            }
    opt_envmap_data = params['envmap.data'].numpy()
    
    print("\nLighting optimized")
    return light_dict, opt_envmap_data


def state2dict(object_state: dict) -> dict:
    """
    Convert object state from gradio to a dictionary for Mitsuba rendering.
    Args:
        object_state (dict): Object state containing "obj_name", "obj_path", "position", "rotation", and "scale".
    Returns:
        object_dict (dict): Dictionary containing the object mesh and its transformation.
    """

    if object_state is None:
        return {}

    object_dict = {}
    print("Convert: ", object_state)

    obj_name = object_state["obj_name"]
    obj_path = object_state["obj_path"]
    position = object_state['position']
    rotation = object_state['rotation']
    scale = object_state['scale']

    mtl_path = extract_mtl_file(obj_path)
    relative_mtl_path = obj_path[:obj_path.rfind('/') + 1] + mtl_path
    texture_paths = extract_texture_paths(relative_mtl_path)
    # TODO: currently only assume one texture image for an object
    relative_texture_path = obj_path[:obj_path.rfind('/')] + texture_paths[0][texture_paths[0].rfind('/'):]
    obj = obj_mesh(obj_path, texture_path=relative_texture_path)

    object_dict[obj_name] = obj
    # compute rotation from euler angles
    rotvec = R.from_euler('xyz', np.array(rotation), degrees=True).as_rotvec(degrees=True)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        axis = np.array([1, 0, 0])
    else:
        axis = rotvec / angle

    print("Insert ", obj_name, " into ", position)

    current=mi.ScalarTransform4f().look_at(
            mi.ScalarPoint3f([0, 0, 0]),
            mi.ScalarPoint3f([0, 0, -1]),
            mi.ScalarPoint3f([0, 1, 0])
        )
    object_dict[obj_name]['to_world'] = current.translate(mi.ScalarPoint3f(position)).rotate(axis=mi.ScalarPoint3f(axis), angle=angle).scale(mi.ScalarPoint3f(scale))

    return object_dict


def insert_object(object_states: list, new_object: dict, position: np.ndarray | list, scale: float=1) -> dict:
    """
    Append a new object to the list of object states.
    Args:
        object_states (list): List of existing object states.
        new_object (dict): New object state containing "obj_name", "obj_path", "position", and "scale".
        position (np.ndarray | list): Position of the new object.
        scale (float): Scale of the new object.
    Returns:
        object_states (list): Updated list of object states.
    """

    if new_object is None:
        return object_states

    print("Object insertion: ", new_object)

    obj_name = new_object["obj_name"]
    obj_path = new_object["obj_path"]

    for i in range(len(object_states)):
        if any(obj_name==obj['obj_name'] for obj in object_states):
            obj_name = obj_name.split('_')[0] + '_' + str(i+2)
        else:
            break

    object_states.append(
        {
            'obj_name': obj_name,
            'obj_path': obj_path,
            'position': position,
            'rotation': [0,0,0],
            'scale': [scale, scale, scale]
        }
    )

    return object_states


def reconstruct_image(model_states: dict, img: np.ndarray, rescale_factor: float, scene_dict: dict, interactive_state: dict, lights_mask: dict, spp: int=48) -> Tuple[None, dict, dict, float]:
    """
    Reconstruct the image using the given model states and input image.
    Args:
        model_states (dict): Dictionary containing MoGe and Intrinsic Decomposition models.
        img (np.ndarray): Input image.
        rescale_factor (float): Rescale factor for the input image.
        scene_dict (dict): Scene dictionary for Mitsuba rendering.
        interactive_state (dict): Interactive state for rendering.
        lights_mask (dict): Mask of the light sources specified by users.
        hr_spp (int): Samples per pixel for high-resolution rendering.
    Returns:
        None: will render reconstructed image later.
        scene_dict (dict): Updated scene dictionary for Mitsuba rendering.
        interactive_state (dict): Updated interactive state for rendering.
        auto_rescale_factor (float): Rescale factor for the scene.
    """
    device = 'cuda'
    threshold = 0.02 # threshold for cutting mesh edges
    sub_scale = 0.5 # scale factor for rendering/optimization
    intrinsic_model = model_states["intrinsic_model"]
    moge_model = model_states["moge_model"]

    # prepare image
    img = img.astype(np.single) / float((2 ** 8) - 1)
    interactive_state["origin_img"] = img
    img = rescale(img, rescale_factor)
    interactive_state["src_img"] = img
    # decompose image into albedo and shading components
    image, albedo, dif_shd = intrinsic_decomposition(intrinsic_model, img)
    interactive_state["albedo"] = albedo
    interactive_state["dif_shd"] = dif_shd

    # rescale image for optimizing lighting conditions
    height, width = image.shape[:2]
    sub_h = math.ceil(height * sub_scale)
    sub_w = math.ceil(width * sub_scale)

    sub_alb = resize(albedo, (sub_h, sub_w))
    sub_dif_shd = resize(dif_shd, (sub_h, sub_w))
    sub_img = resize(image, (sub_h, sub_w))

    # reconstruct geometry from the image
    cam_intrinsics, mask, sky_comp_mask, edge_mask = geometry_reconstruction(moge_model, sub_img, height, width, sub_h, sub_w, albedo, device, threshold)
    interactive_state["sky_comp_mask"] = sky_comp_mask
    interactive_state["edge_mask"] = edge_mask

    print("Geometry constructed")

    # setup the scene dictionary for Mitsuba rendering
    scene_dict = prepare_diffren_scene(cam_intrinsics, sub_h, sub_w, height, width)

    print("scene dictionary constructed")

    # setup optimization target
    np_target = sub_dif_shd * sub_alb # the optimization target is the diffuse image
    target = mi.TensorXf(np_target)

    light_dict, opt_envmap_data = optimize_light(scene_dict, target, lights_mask["layers"][0])
    scene_dict['envmap']['bitmap'] = mi.Bitmap(opt_envmap_data)
    scene_dict = scene_dict | light_dict

    # render shape index and rgb for later use
    inpainted_render, _ = render(scene_dict, interactive_state, 1, spp, 0)
    interactive_state["rgb_wo_obj"] = inpainted_render
    shape_map = render_shape_index(scene_dict, spp)
    print("shape id before object:", np.unique(shape_map))
    print("# of shape id before object:", len(np.unique(shape_map)))
    interactive_state["shape_wo_obj"] = np.unique(shape_map).tolist()

    return None, scene_dict, interactive_state, auto_rescale_factor(mi.load_dict(scene_dict))


def render(scene_dict: dict, interactive_state: dict, sensor_id: int=0, spp: int=4096, diff_compose_weight: float=0, object_states: list=[], save_file: bool=False) -> Tuple[np.ndarray, gr.DownloadButton]:
    """
    Render the scene using Mitsuba and return the rendered image.
    Args:
        scene_dict (dict): Scene dictionary for Mitsuba rendering.
        interactive_state (dict): Interactive state for rendering.
        sensor_id (int): Sensor ID for rendering.
        spp (int): Samples per pixel for rendering.
        diff_compose_weight (float): Weight for differential compositing.
        object_states (list): List of object states to be rendered.
        save_file (bool): Whether to save the rendered image.
    Returns:
        result_img (np.ndarray): Rendered image.
        save_button (gr.DownloadButton): Download button for the rendered image.
    """
    # set integrator to path tracing
    scene_dict['integrator'] = {
        'type': 'path',
        'max_depth': 3
    }

    # update scene dictionary with the current object configurations
    object_dict = {}
    for obj_state in object_states:
        object_dict.update(state2dict(obj_state))
    composed_scene_dict = scene_dict | object_dict
    print("finish composing scene dict")
    # render the scene and conduct gamma correction
    scene = mi.load_dict(composed_scene_dict)
    rendered_insert = mi.render(scene, mi.traverse(scene), sensor=sensor_id, spp=spp)
    # inpainted_render = inpaint_render(rendered_insert, interactive_state["albedo"], interactive_state["dif_shd"], interactive_state["edge_mask"], interactive_state["sky_comp_mask"])
    inpainted_render = np.array(rendered_insert)
    inpainted_render = gamma_correction(interactive_state["origin_img"], inpainted_render)

    inpainted_render = np.clip(inpainted_render, 0, 1)

    result_img = inpainted_render
    src_img = interactive_state["src_img"] # H, W, 3
    original_img = interactive_state["origin_img"]
    original_size = original_img.shape[:2]
    # differential compositing. Here I use depth difference to compute object's mask since mitsuba's aov integrator does not provide correct shape index rendering. Don't know why.
    if diff_compose_weight > 0:
        shape_map = render_shape_index(composed_scene_dict, spp=48)
        print("shape id:", np.unique(shape_map))
        print("# of shape id:", len(np.unique(shape_map)))
        shape_index_list = np.unique(shape_map).tolist()
        shape_index_wo_obj = interactive_state["shape_wo_obj"]
        new_shape_index = list(set(shape_index_list) - set(shape_index_wo_obj))
        obj_mask = np.zeros_like(shape_map)
        for id in new_shape_index:
            obj_mask = np.logical_or(obj_mask, shape_map == id)

        recon_img = interactive_state["rgb_wo_obj"] # H, W, 3
        re_src_img = resize(src_img, recon_img.shape[:2], anti_aliasing=True)

        result_img = differential_compositing(re_src_img, recon_img, inpainted_render, obj_mask, diff_compose_weight)
    # prepare save file
    temp_path = None
    if save_file:
        result_img_fullsize = resize(result_img, original_size, anti_aliasing=True)
        result_img_fullsize = np_to_pil(result_img_fullsize)
        temp_path = os.path.join(temp_dir, "rendered_image.png")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        result_img_fullsize.save(temp_path)

    save_button = None
    if temp_path is not None:
        save_button = gr.DownloadButton(label="Download Result", value=temp_path, visible=True)
    else:
        save_button = gr.DownloadButton(label="Download Result", visible=False)
    return result_img, save_button


def render_depth(scene_dict: dict, spp: int=48) -> np.ndarray:
    """
    Render the depth of the scene using Mitsuba.
    Args:
        scene_dict (dict): Scene dictionary for Mitsuba rendering.
        spp (int): Samples per pixel for rendering.
    Returns:
        depth (np.ndarray): Rendered depth image.
    """
    _scene_dict = scene_dict.copy()
    _scene_dict['integrator'] = {'type': 'aov', 'aovs': 'dd.y:depth'}
    scene = mi.load_dict(_scene_dict)
    rendered_depth = mi.render(scene, sensor=1, spp=spp)
    return np.array(rendered_depth)


def render_shape_index(scene_dict: dict, spp: int=48) -> np.ndarray:
    """
    Render the shape index of the scene using Mitsuba.
    Args:
        scene_dict (dict): Scene dictionary for Mitsuba rendering.
        spp (int): Samples per pixel for rendering.
    Returns:
        shape_index (np.ndarray): Rendered shape index image.
    """
    _scene_dict = scene_dict.copy()
    _scene_dict['integrator'] = {'type': 'aov', 'aovs': 'si:shape_index'}
    scene = mi.load_dict(_scene_dict)
    rendered_shape_index = mi.render(scene, sensor=1, spp=spp)
    return np.array(rendered_shape_index).round(0).astype(np.uint8)


def render_position_and_normal(scene_dict: dict, spp: int=48, sensor: int=1) -> mi.Bitmap:
    """
    Render the position and normal of the scene using Mitsuba.
    Args:
        scene_dict (dict): Scene dictionary for Mitsuba rendering.
        spp (int): Samples per pixel for rendering.
        sensor (int): Sensor ID for rendering.
    Returns:
        aov_out (mi.Bitmap): Rendered position and normal image.
    1. The position is stored in the first three channels (x, y, z).
    2. The normal is stored in the next three channels (nx, ny, nz).
    """
    _scene_dict = scene_dict.copy()
    _scene_dict['integrator'] = {'type': 'aov', 'aovs': 'pos:position, normal:sh_normal'}
    scene = mi.load_dict(_scene_dict)
    aov_out = mi.render(scene, sensor=sensor, spp=spp)
    return aov_out


def differential_compositing(ori_img: np.ndarray, recon_img: np.ndarray, insert_img: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Differential compositing of the input images.
    Args:
        ori_img (np.ndarray): Original image.
        recon_img (np.ndarray): Reconstructed image without object.
        insert_img (np.ndarray): Reconstructed image with object.
        mask (np.ndarray): Mask of the object.
        weight (np.ndarray): Weight for differential compositing.
    Returns:
        compose (np.ndarray): Composed image.
    """
    compose = mask * insert_img + (1 - mask) * (ori_img + weight * (insert_img - recon_img))
    compose = np.clip(compose, 0, 1)
    return compose


def get_position_normal(scene_dict: dict, pixel_coord: tuple, sensor_id: int=1) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """
    Get the position and normal of a pixel in the scene. Using camera ray casting to get the position and normal is faster than render a whole map of position and normal.
    Args:
        scene_dict (dict): Scene dictionary for Mitsuba rendering.
        pixel_coord (tuple): Pixel coordinates (x, y).
        sensor_id (int): Sensor ID for rendering.
    Returns:
        position (np.ndarray | None): Position of the pixel in world space.
        normal (np.ndarray | None): Normal of the pixel in world space.
    """
    x, y = pixel_coord
    scene = mi.load_dict(scene_dict)
    camera = scene.sensors()[sensor_id]
    film = camera.film()
    resolution = film.crop_size()

    import random
    sample1 = random.random()
    sample2 = mi.Point2f(x / resolution[0], y / resolution[1])
    sample3 = mi.Point2f(random.random(), random.random())

    ray, _ = camera.sample_ray(time=0, sample1=sample1, sample2=sample2, sample3=sample3, active=True)
    intersection=scene.ray_intersect(ray)

    if intersection.is_valid():
        position = (np.array(intersection.p)).flatten() * [-1,1,-1]
        normal = (np.array(dr.normalize(intersection.n))).flatten() * [-1,1,-1]
        return position, normal
    else:
        return None, None


def main():

    device = 'cuda'
    threshold = 0.02 # threshold for cutting mesh edges
    sub_scale = 0.5 # scale factor for rendering/optimization

    # prepare image
    img = load_from_url('https://images.unsplash.com/photo-1723642613951-8f17ad33554a')
    img = rescale(img, 0.4)

    image, albedo, dif_shd = intrinsic_decomposition(img)
    # rescale image for optimizing lighting conditions
    height, width = image.shape[:2]
    sub_h = math.ceil(height * sub_scale)
    sub_w = math.ceil(width * sub_scale)

    sub_alb = resize(albedo, (sub_h, sub_w))
    sub_dif_shd = resize(dif_shd, (sub_h, sub_w))
    sub_img = resize(image, (sub_h, sub_w))

    cam_intrinsics, mask, sky_comp_mask, edge_mask = geometry_reconstruction(sub_img, height, width, sub_h, sub_w, albedo, device, threshold)

    grid_size = 4 # size of point light grid (grid_size x grid_size)
    margin = 50 # margin around the image for the grid
    scene_dict = prepare_diffren_scene(cam_intrinsics, sub_h, sub_w, height, width, grid_size, margin)

    # setup optimization target
    np_target = sub_dif_shd * sub_alb # the optimization target is the diffuse image

    target = mi.TensorXf(np_target)

    optimized_params = optimize_light(mi.load_dict(scene_dict), target, mask, grid_size)

    for key in scene_dict.keys():
        if key.find("pointlight") != -1:
            scene_dict[key] = {
                'type': 'point',
                'position': list(np.asarray(optimized_params[key + '.position']).flatten()),
                'intensity': {
                    'type': 'rgb',
                    'value': list(np.asarray(optimized_params[key + '.intensity.value']).flatten())
                }
            }

    # It seems like optimizer also refines envmap?
    # I'm not sure if we should include this:
    opt_envmap_data = optimized_params['envmap.data'].numpy()
    scene_dict['envmap']['bitmap'] = mi.Bitmap(opt_envmap_data)

    # insert object
    insert_object(scene_dict, "teapot", ply_mesh("./assets/teapot-small.ply"))

    # show the original image along side our rendered reconstruction
    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)
    rendered = mi.render(scene, params, sensor=1, spp=4096)
    inpainted_render = inpaint_render(rendered, albedo, dif_shd, edge_mask, sky_comp_mask)
    show([(dif_shd * albedo), inpainted_render])
    plt.show()


if __name__ == '__main__':
    main()