import gradio as gr
from scene_recover import *
import mitsuba as mi
import numpy as np
import drjit


example_image_paths = ["assets/tree/image.jpg", "assets/bag/image.jpg", "assets/bear/image.jpg", "assets/cage/image.jpg", \
                       "assets/chair/image.jpg", "assets/desk/image.jpg", "assets/lamp/image.jpg", "assets/plant/image.jpg", \
                       "assets/shelf/image.jpg", "assets/vase/image.jpg"]

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


def get_select_obj(evt: gr.SelectData, interactive_state: gr.State) -> gr.State:
    select_index = evt.index
    interactive_state["src_obj_path"] = example_image_paths[select_index].replace("image.jpg", "normalized_model.obj")
    interactive_state["src_obj_name"] = example_image_paths[select_index].split('/')[1]
    return interactive_state


def main():
    initialize()

    with gr.Blocks(theme=gr.themes.Base()) as gui:
        scene_dict = gr.State()
        interactive_state = gr.State({
            "src_img": None,
            "src_obj_path": None,
            "src_obj_name": None,
            "albedo": None,
            "dif_shd": None,
            "edge_mask": None,
            "sky_comp_mask": None,
            "depth_wo_obj": None,
            "rgb_wo_obj": None,
            "rgb_w_obj": None,
        })

        gr.HTML(
            """
            <div style="text-align: center; font-weight: bold; font-size: 2em; margin-bottom: 5px">
            Image Composing
            </div>
            <div align="center">
            Insert 3d object into 2d image
            </div>
            """)


        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Input Image"):
                    src_image_path = gr.Image()
                    btn_1 = gr.Button("Start", interactive=False)
                    with gr.Accordion("Advanced Options", open=False):
                        gen_scale = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.05, label="Scaling Factor", info="An image will be scaled down for faster process")
                    
                with gr.Tab("Object Insertion", elem_id='sidebar'):
                    with gr.Group():
                        gr.HTML("""<div align="center">Translation</div>""")
                        translation_x = gr.Slider(minimum=-10, maximum=10, value=0, label="translation_x")
                        translation_y = gr.Slider(minimum=-10, maximum=10, value=0, label="translation_y")
                        translation_z = gr.Slider(minimum=-10, maximum=10, value=0, label='translation_z')

                    with gr.Group():
                        gr.HTML("""<div align="center">Rotation (degree)</div>""")
                        rotation_x = gr.Slider(minimum=-180, maximum=180, value=0, label="rotation_x")
                        rotation_y = gr.Slider(minimum=-180, maximum=180, value=0, label="rotation_y")
                        rotation_z = gr.Slider(minimum=-180, maximum=180, value=0, label='rotation_z')
                    with gr.Group():
                        gr.HTML("""<div align="center">Scale</div>""")
                        scale_uni = gr.Checkbox(label='Uniform', value=True, interactive=True)
                        scale_x = gr.Slider(minimum=0.1, maximum=10, value=1, label="scale_x", step=0.1)
                        scale_y = gr.Slider(minimum=0.1, maximum=10, value=1, label="scale_y", step=0.1)
                        scale_z = gr.Slider(minimum=0.1, maximum=10, value=1, label="scale_z", step=0.1)
                        scale_x.release(fn=lambda x,y,z,b: [x,x,x] if b else [x,y,z], inputs=[scale_x, scale_y, scale_z, scale_uni], outputs=[scale_x, scale_y, scale_z])
                        scale_y.release(fn=lambda x,y,z,b: [y,y,y] if b else [x,y,z], inputs=[scale_x, scale_y, scale_z, scale_uni], outputs=[scale_x, scale_y, scale_z])
                        scale_z.release(fn=lambda x,y,z,b: [z,z,z] if b else [x,y,z], inputs=[scale_x, scale_y, scale_z, scale_uni], outputs=[scale_x, scale_y, scale_z])
                    with gr.Group():
                        gr.HTML("""<div align="center">Differential Compositing Weight</div>""")
                        compose_weight = gr.Slider(minimum=0.01, maximum=2, value=1.0, label="weight", step=0.01)

                    btn_2 = gr.Button("Insert", interactive=False)
            with gr.Column(scale=2):
                res_image = gr.Image()
                btn_3 = gr.Button("Render")
            
            with gr.Column(scale=1):
                with gr.Tab('Input Image Presets'):
                    with gr.Column():
                        gr.Examples(
                            label='Select an image from presets',
                            examples=[
                                'https://images.unsplash.com/photo-1723642613951-8f17ad33554a',
                                'https://images.unsplash.com/photo-1737103515275-ebc1b6934c13',
                            ],
                            inputs=src_image_path,
                        )
                with gr.Tab('3D Object Presets'):
                    gr.Markdown("### Select a 3D object by clicking its image:")
                    gallery = gr.Gallery(
                        value=example_image_paths,  # Single images per row
                        label="Example Objects",
                        show_label=False,
                        columns=2,  # Display images in 2 columns
                        height="auto",
                        object_fit="contain",
                        allow_preview=False
                    )

        src_image_path.change(fn=lambda x: gr.Button("Start", interactive=False, variant='secondary') if x is None else gr.Button("Start", interactive=True, variant='primary'), inputs=src_image_path, outputs=btn_1)
        btn_1.click(
                fn=generate_3D_mesh, 
                inputs=[src_image_path, gen_scale, scene_dict, interactive_state], 
                outputs=[res_image, scene_dict, interactive_state]
            ).then(
                lambda: gr.update(interactive=True, variant='primary'), 
                outputs=btn_2
            )

        gallery.select(get_select_obj, inputs=interactive_state, outputs=interactive_state)
        btn_2.click(fn=insert_object, inputs=[scene_dict, interactive_state, translation_x, translation_y, translation_z, \
                                              rotation_x, rotation_y, rotation_z, scale_x, scale_y, scale_z, compose_weight], outputs=[scene_dict, res_image])
        
        btn_3.click(fn=render, inputs=[scene_dict, interactive_state], outputs=res_image,)
            

    gui.launch()

if __name__ == '__main__':
    main()