import gradio as gr
from scene_recover import *
import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np
import drjit

def initialize():
    # Couldn't figure out how to run mitsuba.load_dict() without this:
    mi.set_variant('cuda_ad_rgb')
    mesh_dict = ply_mesh("./assets/teapot-small.ply")
    mesh = mi.load_dict(mesh_dict)
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'aov',
            'aovs': 'pos:position, normal:sh_normal',
        },
        'sensor1': {
            'type': 'perspective',
            'fov': float(1), 
            'fov_axis' : 'y',
            'to_world': mi.ScalarTransform4f().look_at(
                mi.ScalarPoint3f([0, 0, 0]),  # camera at origin
                mi.ScalarPoint3f([0, 0, -1]), # looking down the -z axis (?) not sure about mitsuba conventions
                mi.ScalarPoint3f([0, 1, 0])
            ),
            'film': {
                'type': 'hdrfilm',
                'width': 1,
                'height': 1
            }
        },
        'envmap': {
            'type': 'envmap',
            'bitmap' : mi.Bitmap(np.ones((128, 256, 3)) * 1),
        },
        'scene_mesh': mesh
    }
    scene = mi.load_dict(scene_dict)
    scene_dict['integrator'] = {
        'type': 'path',
        'max_depth': 3
    }

    scene_dict['sampler'] = {
        'type': 'orthogonal',
    }
    scene_dict['point_light'] = {
        'type': 'point',
        'position': drjit.cuda.Float64(0,0,0),
        'intensity': {
            'type': 'rgb',
            'value': [0.01, 0.01, 0.01]
        }
    }
    scene = mi.load_dict(scene_dict)

with gr.Blocks(theme=gr.themes.Soft()) as gui:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("Input Image"):
                src_image_path = gr.Textbox(value='https://images.unsplash.com/photo-1723642613951-8f17ad33554a', label="Image URL")
                src_image = gr.Image(src_image_path.value)
                src_image_path.change(fn=lambda x:x, inputs=src_image_path, outputs=src_image)
                btn_1 = gr.Button("Start")
                
            with gr.Tab("Object Insertion"):
                with gr.Group():
                    gr.Markdown("Position")
                    location_x = gr.Slider(minimum=1, maximum=10, label="x")
                    location_y = gr.Slider(minimum=1, maximum=10, label="y")
                    location_z = gr.Slider(minimum=1, maximum=10, label='z')

                with gr.Group():
                    gr.Markdown("Rotation")
                    rotation_x = gr.Slider(minimum=1, maximum=10, label="x")
                    rotation_y = gr.Slider(minimum=1, maximum=10, label="y")
                    rotation_z = gr.Slider(minimum=1, maximum=10, label='z')
                with gr.Group():
                    gr.Markdown("Scale")
                    scale_x = gr.Slider(minimum=1, maximum=10, label="x")
                    scale_y = gr.Slider(minimum=1, maximum=10, label="y")
                    scale_z = gr.Slider(minimum=1, maximum=10, label='z')
        with gr.Column(scale=2):
            res_image = gr.Image()
    
    btn_1.click(fn=generate_3D_mesh, inputs=src_image_path, outputs=res_image)

def main():
    initialize()
    gui.launch()

if __name__ == '__main__':
    main()