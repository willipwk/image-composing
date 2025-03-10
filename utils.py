import trimesh
import cv2
import numpy as np
import mitsuba as mi
import drjit as dr
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
