import sys
import os

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

import matplotlib.pyplot as plt

import trimesh
import open3d as o3d
from typing import Tuple
from utils import save_ply, mse, inpaint_render, ply_mesh, str2float_tuple


def intrinsic_decomposition(img) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # intrinsic decomposition
    intrinsic_model = load_models('v2')

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
    show([image, albedo, dif_shd])
    return image, albedo, dif_shd


def geometry_reconstruction(image: np.ndarray, height: int, width: int, sub_h: int, sub_w: int, albedo: np.ndarray, device: str, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # recover geometry
    moge_model = MoGeModel.from_pretrained('Ruicheng/moge-vitl').to('cuda').eval()
    image_tensor = torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1)
    output = moge_model.infer(image_tensor ** (1/2.2))
    points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
    points = points.clip(max=2**32)

    points = resize(points, (height, width))
    depth = resize(depth, (height, width))
    mask = resize(mask, (height, width))

    sky_mask = ~mask
    sky_comp_mask = sky_mask[..., None].astype(np.single)
    sub_sky_msk = resize(sky_comp_mask, (sub_h, sub_w))

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


def prepare_diffren_scene(intrinsics: np.ndarray, sub_h: int, sub_w: int, height: int, width: int, grid_size: int, margin: int):
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
            'type': 'aov',
            'aovs': 'pos:position, normal:sh_normal',
        },
        'sensor1': cam_dict1,
        'sensor2': cam_dict2,
        'envmap': {
            'type': 'envmap',
            'bitmap' : mi.Bitmap(np.ones((128, 256, 3)) * 1),
        },
        'scene_mesh': mesh
    }

    scene = mi.load_dict(scene_dict)

    # # render the position uv for each pixel
    aov_out = mi.render(scene, spp=1)
    positions = aov_out[:, :, :3]
    normals = aov_out[:, :, 3:6]

    # create a grid of evenly spaced values in image space (with some margin)
    im_grid = np.meshgrid(np.linspace(margin, sub_w-(margin), grid_size), np.linspace(margin, sub_h-(margin), grid_size))

    pl_count = 0
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


    # we render the scene with the path tracer integrator, 
    # I set the max depth to 2 to speed up the render, but this can be experimented with
    scene_dict['integrator'] = {
        'type': 'path',
        'max_depth': 3
    }

    scene_dict['sampler'] = {
        'type': 'orthogonal',
    }

    scene = mi.load_dict(scene_dict)
    scene_params = mi.traverse(scene)

    test_render = mi.render(scene, sensor=1, spp=64)

    show(test_render)

    return scene_dict


def optimize_light(scene, target, mask, grid_size: int):
    # we optimize the light positions and intensities along side the environment map
    params = mi.traverse(scene)

    keys = []
    for i in range(grid_size ** 2):
        keys.append(f'pointlight_{i}.position')
        keys.append(f'pointlight_{i}.intensity.value')
    keys.append('envmap.data')
    learning_rate = 0.01
    opt = mi.ad.Adam(lr=learning_rate, mask_updates=True, beta_1=0.8)
    for k in keys:
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
            mask=mask
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
    return params

def insert_object(scene_dict, obj_name, obj):
    scene_dict[obj_name] = obj

    scene = mi.load_dict(scene_dict)
    scene_params = mi.traverse(scene)

    while(True):
        #print(f"Current Transform: {scene_dict['teapot']['to_world']}")
        command = input("type \"trans\" to translate, \"rotate\" to rotate, \"exit\" to stop: ")
        current = scene_dict[obj_name]['to_world']
        print(current)

        if command == "trans":
            command = input("input translate value separated by comma: ")
            locations = str2float_tuple(command)
            if locations==None: continue
            scene_dict[obj_name]['to_world'] = current.translate(mi.ScalarPoint3f(locations))

        elif command == "rotate":
            command = input("input axis and angle separated by comma (e.g.) 0.0,1.0,0.0,90: ")
            rotation = str2float_tuple(command, 4)
            if rotation==None: continue
            scene_dict[obj_name]['to_world'] = current.rotate(axis=mi.ScalarPoint3f(rotation[:3]), angle=rotation[3])

        elif command == "exit":
            break
        else:
            print("Invalid input: ", command)
        

        scene = mi.load_dict(scene_dict)
        show(mi.render(scene, mi.traverse(scene), sensor=1, spp=64))
        plt.show()
    return scene_dict


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