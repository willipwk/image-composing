import trimesh
import cv2
import numpy as np
import mitsuba as mi
import drjit as dr
import os
from chrislib.general import invert, uninvert


def save_ply(path, vertices, faces, vertex_colors):
    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        process=False
    ).export(path)


def writeexr(I, path):
    EXR_OPTIONS=[cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PIZ]
    cv2.imwrite(path, cv2.cvtColor(np.float32(I), cv2.COLOR_RGB2BGR), EXR_OPTIONS)
    

def load_exr(path):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return img[..., ::-1]


def mse(image, target, mask=None):
    return dr.mean(dr.square(image - target))


def inpaint_render(render, albedo, dif_shd, edge_mask, sky_mask, comp_sky=True):
    rendered_shading = np.array(render / albedo.clip(0.0001))

    # inpaint the rendered shading to fill in the depth edges
    tm_shd = 1.0 - invert(rendered_shading)
    tm_shd = np.nan_to_num(tm_shd, nan=0.0, posinf=0.0, neginf=0.0)
    
    src = (tm_shd * 255.).astype(np.uint8)

    blend_mask = ((~edge_mask).astype(np.single) * 255.).astype(np.uint8)[:, :, None]
    kernel = np.ones((5, 5), np.uint8)
    blend_mask = cv2.dilate(blend_mask, kernel, iterations = 2)

    inpainted = cv2.inpaint(src, blend_mask, 3, cv2.INPAINT_TELEA)
    inpainted = inpainted.astype(np.single) / 255.

    inpainted = uninvert(1 - inpainted)

    combine_mask = 1 - (blend_mask.astype(np.float32) / 255)[..., None]
    filled = (1 - combine_mask) * inpainted + (combine_mask * rendered_shading)

    inpainted_render = (filled * albedo)

    dif_img = albedo * dif_shd

    if comp_sky:
        final_render = (sky_mask * dif_img) + (1 - sky_mask) * inpainted_render
    else:
        final_render = inpainted_render

    return final_render

def ply_mesh(mesh_path, scale_factor=1, transform=[0.0, 0.0, 0.0], bsdf=None):
    shape_dict = {
        'type': 'ply', 
        'filename': mesh_path
    }

    if transform:
        shape_dict['to_world'] = mi.ScalarTransform4f().look_at(
        mi.ScalarPoint3f(transform),  # camera at origin
        mi.ScalarPoint3f([0, 0, -1]), 
        mi.ScalarPoint3f([0, 1, 0])
    )
    
    if bsdf:
        shape_dict['bsdf'] = bsdf
    else:
        shape_dict['bsdf'] = {
            'type': 'diffuse',
            'reflectance':
            {
                'type': 'mesh_attribute',
                'name': 'vertex_color'
            }
        }
    
    return shape_dict


def ply_mesh_texture(mesh_path, scale_factor=1, transform=[0.0, 0.0, 0.0], bsdf=None):
    shape_dict = {
        'type': 'ply', 
        'filename': mesh_path
    }

    if transform:
        shape_dict['to_world'] = mi.ScalarTransform4f().look_at(
        mi.ScalarPoint3f(transform),  # camera at origin
        mi.ScalarPoint3f([0, 0, -1]), 
        mi.ScalarPoint3f([0, 1, 0])
    )
    
    if bsdf:
        shape_dict['bsdf'] = bsdf
    else:
        shape_dict['bsdf'] = {
            'type': 'diffuse',
            'reflectance': 
            {
                'type': 'bitmap',
                'filename': 'assets/tree/texture.png'
            }
        }
    
    return shape_dict


def obj_mesh(mesh_path, scale_factor=1, transform=[0.0, 0.0, 0.0], bsdf=None, texture_path=None):
    shape_dict = {
        'type': 'obj', 
        'filename': mesh_path
    }

    if transform:
        shape_dict['to_world'] = mi.ScalarTransform4f().look_at(
        mi.ScalarPoint3f(transform),  # camera at origin
        mi.ScalarPoint3f([0, 0, -1]), 
        mi.ScalarPoint3f([0, 1, 0])
    )
    
    if bsdf:
        shape_dict['bsdf'] = bsdf
    else:
        if texture_path is not None:
            shape_dict['bsdf'] = {
                'type': 'diffuse',
                'reflectance': 
                {
                    'type': 'bitmap',
                    'filename': texture_path
                }
            }
        else:
            shape_dict['bsdf'] = {
                'type': 'diffuse',
                'reflectance':
                {
                    'type': 'mesh_attribute',
                    'name': 'vertex_color'
                }
            }
    
    return shape_dict


def extract_mtl_file(obj_file_path):
    """Extract the associated MTL file from the OBJ file."""
    mtl_filename = None
    with open(obj_file_path, 'r') as obj_file:
        for line in obj_file:
            if line.lower().startswith('mtllib'):
                mtl_filename = line.strip().split()[1]
                break
    return mtl_filename

def extract_texture_paths(mtl_file_path):
    """Extract texture paths from the MTL file."""
    texture_paths = []
    texture_keywords = ['map_Kd', 'map_Ka', 'map_Ks', 'map_Bump', 'bump', 'disp', 'decal', 'map_d']

    if not os.path.exists(mtl_file_path):
        print(f"MTL file not found: {mtl_file_path}")
        return texture_paths

    with open(mtl_file_path, 'r') as mtl_file:
        for line in mtl_file:
            print("line:", line)
            for keyword in texture_keywords:
                if line.startswith(keyword):
                    print("line.strip:", line.strip())
                    parts = line.strip().split()
                    print("parts:", parts)
                    if len(parts) > 1:
                        texture_path = parts[1]
                        texture_paths.append(texture_path)
    return texture_paths


def str2float_tuple(input, size=3):
    """
    Converts a string of three floats separated by commas into a tuple of floats.
    Returns None if size of tuple does not match
    """
    try:
        float_list = [float(x) for x in input.split(',')]
        if len(float_list) == size:
            return tuple(float_list)
        else:
            return None
    except ValueError:
        return None