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
    texture_mesh_dict = obj_mesh("./assets/tree/normalized_model.obj", texture_path="./assets/tree/texture.png")
    texture_mesh = mi.load_dict(texture_mesh_dict)
    # obj_dict = obj_mesh("./assets/tree/normalized_model.obj")
    # obj = mi.load_dict(obj_dict)
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
        # 'scene_mesh': mesh
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


def get_slider_number(label, slider, interactive_state):
    interactive_state[label] = slider

    return interactive_state


def main():
    initialize()

    with gr.Blocks(theme=gr.themes.Soft()) as gui:
        scene_dict = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Input Image"):
                    # src_image_path = gr.Textbox(value='https://images.unsplash.com/photo-1723642613951-8f17ad33554a', label="Image URL")
                    src_image_path = gr.File()
                    src_image = gr.Image(src_image_path.value)
                    src_image_path.change(fn=lambda x:x, inputs=src_image_path, outputs=src_image)
                    btn_1 = gr.Button("Start")
                    
                with gr.Tab("Object Insertion"):
                    src_obj_path = gr.Textbox(value="./assets/bag/normalized_model.obj", label="Object Path")
                    # src_texture_path = gr.Textbox(value="./assets/tree/texture.png", label="Texture Path")
                    src_obj_name = gr.Textbox(value="bag", label="Object Name")
                    with gr.Group():
                        gr.Markdown("Translation")
                        translation_x = gr.Slider(minimum=-10, maximum=10, value=0, label="translation_x")
                        translation_y = gr.Slider(minimum=-10, maximum=10, value=0, label="translation_y")
                        translation_z = gr.Slider(minimum=-10, maximum=10, value=0, label='translation_z')

                    with gr.Group():
                        gr.Markdown("Rotation (degree)")
                        rotation_x = gr.Slider(minimum=-180, maximum=180, value=0, label="rotation_x")
                        rotation_y = gr.Slider(minimum=-180, maximum=180, value=0, label="rotation_y")
                        rotation_z = gr.Slider(minimum=-180, maximum=180, value=0, label='rotation_z')
                    with gr.Group():
                        gr.Markdown("Scale")
                        scale_x = gr.Slider(minimum=0.1, maximum=10, value=1, label="scale_x")
                        scale_y = gr.Slider(minimum=0.1, maximum=10, value=1, label="scale_y")
                        scale_z = gr.Slider(minimum=0.1, maximum=10, value=1, label="scale_z")
                    btn_2 = gr.Button("Insert")
            with gr.Column(scale=2):
                res_image = gr.Image()
        btn_1.click(fn=generate_3D_mesh, inputs=[src_image_path, scene_dict], outputs=[res_image, scene_dict])
        btn_2.click(fn=insert_object, inputs=[scene_dict, src_obj_name, src_obj_path, translation_x, translation_y, translation_z, \
                                              rotation_x, rotation_y, rotation_z, \
                                              scale_x, scale_y, scale_z], outputs=[scene_dict, res_image])

    gui.launch()

if __name__ == '__main__':
    main()