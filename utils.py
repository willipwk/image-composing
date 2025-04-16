import trimesh
import cv2
import numpy as np
import mitsuba as mi
import drjit as dr
import os
from chrislib.general import invert, uninvert


def save_ply(path: str, vertices: np.ndarray, faces: np.ndarray, vertex_colors: np.ndarray):
    """
    Save a mesh to a PLY file.
    Args:
        path (str): Path to save the PLY file.
        vertices (np.ndarray): Array of vertex positions.
        faces (np.ndarray): Array of face indices.
        vertex_colors (np.ndarray): Array of vertex colors.
    """
    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        process=False
    ).export(path)
    

def mse(image, target):
    """
    Calculate the Mean Squared Error (MSE) between two images.
    Args:
        image (mi.TensorXf): The first image.
        target (mi.TensorXf): The second image.
    Returns:
        mi.TensorXf: The MSE value.
    """
    return dr.mean(dr.square(image - target))


def inpaint_render(render: mi.Bitmap, albedo: np.ndarray, dif_shd: np.ndarray, edge_mask: np.ndarray, sky_mask: np.ndarray, comp_sky: bool=True) -> np.ndarray:
    """
    Inpaints the rendered shading to fill in the depth edges.
    Args:
        render (mi.Bitmap): The rendered image.
        albedo (np.ndarray): The albedo image.
        dif_shd (np.ndarray): The diffuse shading image.
        edge_mask (np.ndarray): The edge mask.
        sky_mask (np.ndarray): The sky mask.
        comp_sky (bool): Whether to composite the sky or not.
    Returns:
        np.ndarray: The final render.
    """
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


def ply_mesh(mesh_path: str, scale_factor: float=1, transform: list=[0.0, 0.0, 0.0], bsdf: dict=None) -> dict:
    """
    Load a PLY mesh and return its shape dictionary.
    Args:
        mesh_path (str): Path to the PLY file.
        scale_factor (float): Scale factor for the mesh.
        transform (list): Transformation matrix for the mesh.
        bsdf (dict): BSDF properties.
    Returns:
        dict: Shape dictionary for the mesh.
    """
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


def obj_mesh(mesh_path: str, scale_factor: float=1, transform: list=[0.0, 0.0, 0.0], bsdf: dict=None, texture_path: str=None) -> dict:
    """
    Load an OBJ mesh and return its shape dictionary.
    Args:
        mesh_path (str): Path to the OBJ file.
        scale_factor (float): Scale factor for the mesh.
        transform (list): Transformation matrix for the mesh.
        bsdf (dict): BSDF properties.
        texture_path (str): Path to the texture file.
    Returns:
        dict: Shape dictionary for the mesh.
    """
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


def extract_mtl_file(obj_file_path: str):
    """
    Extract the associated MTL file from the OBJ file.
    Args:
        obj_file_path (str): Path to the OBJ file.
    Returns:
        str: Path to the MTL file.
    """
    mtl_filename = None
    with open(obj_file_path, 'r') as obj_file:
        for line in obj_file:
            if line.lower().startswith('mtllib'):
                mtl_filename = line.strip().split()[1]
                break
    return mtl_filename


def extract_texture_paths(mtl_file_path: str):
    """
    Extract texture paths from the MTL file.
    Args:
        mtl_file_path (str): Path to the MTL file.
    Returns:
        list: List of texture paths.
    """
    texture_paths = []
    texture_keywords = ['map_Kd', 'map_Ka', 'map_Ks', 'map_Bump', 'bump', 'disp', 'decal', 'map_d']

    if not os.path.exists(mtl_file_path):
        print(f"MTL file not found: {mtl_file_path}")
        return texture_paths

    with open(mtl_file_path, 'r') as mtl_file:
        for line in mtl_file:
            for keyword in texture_keywords:
                if line.startswith(keyword):
                    parts = line.strip().split()
                    if len(parts) > 1:
                        texture_path = parts[1]
                        texture_paths.append(texture_path)
    return texture_paths


def gamma_correction(original_image: np.ndarray, reconstructed_image: np.ndarray) -> np.ndarray:
    """
    Adjusts the gamma of the reconstructed image to match the brightness of the original image.
    Args:
        original_image (np.ndarray): The original image.
        reconstructed_image (np.ndarray): The reconstructed image.
    Returns:
        corrected_image (np.ndarray): The gamma-corrected image.
    """

    corrected_image = np.zeros_like(reconstructed_image)

    for channel in range(reconstructed_image.shape[-1]):
        mean_original = np.mean(original_image[..., channel])
        mean_reconstructed = np.mean(reconstructed_image[..., channel])

        if mean_reconstructed > 0:
            gamma = np.log(mean_original) / np.log(mean_reconstructed)
            corrected_image[..., channel] = np.power(reconstructed_image[..., channel], gamma)
        else:
            corrected_image[..., channel] = reconstructed_image[..., channel]
    
    return np.clip(corrected_image, 0, 1)


def auto_rescale_factor(scene, margin: float=0.1):
    """
    Returns rescale factor of the object to fit within scene. Assuming that the object is normalized.
    Args:
        scene (mi.Scene): The scene containing the object.
        margin (float): Margin to add around the object.
    Returns:
        float: The rescale factor.
    """
    scene_bounding_box = scene.bbox()

    edge_lengths = []
    for i in range(3):
        edge_lengths.append(scene_bounding_box.max[i] - scene_bounding_box.min[i])

    return margin * max(edge_lengths)