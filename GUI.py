import gradio as gr
from scene_recover import *
import mitsuba as mi
import numpy as np
import drjit
import json

from chrislib.general import rescale
def auto_sacle(img, factor):
    img = img.astype(np.single) / float((2 ** 8) - 1)
    img = rescale(img, factor)
    img = rescale(img, 0.5)
    shape = img.shape
    print("size: ", shape[:2])
    return img, shape[:2]


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
    scene_dict['new_sphere'] = {
        "type": "sphere",
        "radius": 0.1,
        "to_world": mi.ScalarTransform4f().translate(mi.ScalarPoint3f([-0.8, 0, -2.88])),
        "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": [1, 0, 0]
                    }
                },
        }
    scene = mi.load_dict(scene_dict)


def get_select_obj(evt: gr.SelectData, selected_obj_state: gr.State) -> gr.State:
    select_index = evt.index
    selected_obj_state['obj_name'] = example_image_paths[select_index].split('/')[1]
    selected_obj_state['obj_path'] = example_image_paths[select_index].replace("image.jpg", "normalized_model.obj")

    return selected_obj_state

def get_pixel_coord(evt: gr.SelectData):
    return evt.index

def main():
    # need to load some api before really use mitsuba
    initialize()
    # load moge and intrinsics decomposition model
    moge_model = MoGeModel.from_pretrained('Ruicheng/moge-vitl').to('cuda').eval()
    intrinsic_model = load_models('v2')

    # GUI
    with gr.Blocks(theme=gr.themes.Base()) as gui:
        scene_dict = gr.State() # geometry of the input
        # this is for intermediate variables
        interactive_state = gr.State({
            "origin_img": None,
            "src_img": None,
            "albedo": None,
            "dif_shd": None,
            "edge_mask": None,
            "sky_comp_mask": None,
            "depth_wo_obj": None,
            "rgb_wo_obj": None,
            "rgb_w_obj": None,
            "aov_out": None,
            "shape_wo_obj": None,
            "shape_w_obj": None,
        })

        auto_rescale_factor = gr.State() # rescale 3D object when inserted
        object_states = gr.State([])    # list of object state {"obj_name", "obj_path", "position", "rotation", "scale"}
        selected_obj_state = gr.State({})   # selected object information {"obj_name", "obj_path"}
        model_states = gr.State({"moge_model": moge_model, "intrinsic_model": intrinsic_model}) # to pass torch model to function via gradio

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
            # left column for input image
            with gr.Column(scale=1):
                with gr.Tabs() as input_tab:
                    with gr.TabItem("Input Image", id=0):
                        src_image_path = gr.Image()
                        with gr.Accordion("Advanced Options", open=False):
                            gen_scale = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.05, label="Scaling Factor", info="An image will be scaled down for faster process")
                            hr_spp = gr.Slider(minimum=512, maximum=4096, value=1024, step=512, label="High Quality spp")
                            preview_spp = gr.Slider(minimum=1, maximum=64, value=48, step=1, label="Preview spp")
                        btn_1 = gr.Button("Start", interactive=False)
                        
                        gr.Examples(
                            label='Select an image from presets',
                            examples=[
                                'https://images.unsplash.com/photo-1723642613951-8f17ad33554a',
                                'https://images.unsplash.com/photo-1737103515275-ebc1b6934c13',
                            ],
                            inputs=src_image_path,
                        )

                    with gr.TabItem("Input 3D Object", id=1):
                        gr.Markdown("### Insert a 3D object from presets")
                        gr.Markdown("After selecting, click on the desired location in the image to insert.")

                        gallery = gr.Gallery(
                            value=example_image_paths,  # Single images per row
                            label="Example Objects",
                            show_label=False,
                            columns=2,  # Display images in 2 columns
                            height="auto",
                            object_fit="contain",
                            allow_preview=False
                        )
                        gallery.select(get_select_obj, inputs=selected_obj_state, outputs=selected_obj_state)
            # middle column for rendering GUI
            with gr.Column(scale=2):
                with gr.Tabs() as render_tab:
                    with gr.TabItem("Rendered Image", id=0):
                        res_image = gr.Image()
                        btn_3 = gr.Button("Render", variant='primary')
                        btn_4 = gr.DownloadButton(label="Save Image", visible=False)
                        compose_weight = gr.Slider(minimum=0, maximum=2, value=1, label="Differential Compositing Weight", step=0.01)

                    with gr.TabItem("Lighting Estimation", id=1):
                        gr.Markdown("### Let's estimate lighting environment in the image")
                        gr.Markdown("The geometry of the image has been constructed. Please mark the light sources you identify on the left image.\nIf no input was given, light sources are inferred automatically.")
                        temp_image = gr.ImageEditor(interactive=True, layers=False, sources=None)
                        temp = gr.State()
                        temp_size = gr.State()
                        src_image_path.change(auto_sacle, [src_image_path, gen_scale], [temp, temp_size]).then(
                            lambda x, y: {'background':x, 'composite':x, 'layers':[np.zeros((y[0], y[1], 4), np.uint8)]}, [temp, temp_size], temp_image
                        )
                
                btn_3.click(render, inputs=[scene_dict, interactive_state, gr.State(1), hr_spp, compose_weight, object_states, gr.State(True)], outputs=[res_image, btn_4])

                coord2D = gr.State()
                coords3D = gr.State()
                normal3D = gr.State()
            # right column for object manipulation
            with gr.Column(scale=1):
                btn_2 = gr.Button("Update", interactive=False)

                # list of 3d objects in the scene
                @gr.render(inputs=[object_states])
                def show_objects(objects):
                    def updateState(new_value, obj_index, type, xyz_index):
                        objects[obj_index][type][xyz_index] = new_value
                    def removeObj(obj_index):
                        objects.pop(obj_index)
                        return objects
                    
                    print('there are ', len(objects), ' objects')
                    type_position = gr.State('position')
                    type_rotation = gr.State('rotation')
                    type_scale = gr.State('scale')
                    x_index = gr.State(0)
                    y_index = gr.State(1)
                    z_index = gr.State(2)
                    
                    for i in range(len(objects)):
                        obj_index = gr.Number(i, visible=False)
                        obj = objects[i]
                        with gr.Accordion(obj['obj_name']):
                            with gr.Group():
                            # gr.HTML("""<div align="center">Position</div>""")
                                pos = obj['position']
                                pos_x_slider = gr.Slider(minimum=-5, maximum=5, value=pos[0], step=0.01, label="position_x")
                                pos_y_slider = gr.Slider(minimum=-5, maximum=5, value=pos[1], step=0.01, label="position_y")
                                pos_z_slider = gr.Slider(minimum=-5, maximum=5, value=pos[2], step=0.01, label='position_z')
                                pos_x_slider.change(updateState, inputs=[pos_x_slider, obj_index, type_position, x_index]).then(render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4])
                                pos_y_slider.change(updateState, inputs=[pos_y_slider, obj_index, type_position, y_index]).then(render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4])
                                pos_z_slider.change(updateState, inputs=[pos_z_slider, obj_index, type_position, z_index]).then(render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4])

                            # gr.HTML("""<div align="center">Rotation (degree)</div>""")
                            with gr.Group():
                                rotation = obj['rotation']
                                rotation_x_slider = gr.Slider(minimum=-180, maximum=180, value=rotation[0], label="rotation_x")
                                rotation_y_slider = gr.Slider(minimum=-180, maximum=180, value=rotation[1], label="rotation_y")
                                rotation_z_slider = gr.Slider(minimum=-180, maximum=180, value=rotation[2], label='rotation_z')
                                rotation_x_slider.change(updateState, inputs=[rotation_x_slider, obj_index, type_rotation, x_index]).then(render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4])
                                rotation_y_slider.change(updateState, inputs=[rotation_y_slider, obj_index, type_rotation, y_index]).then(render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4])
                                rotation_z_slider.change(updateState, inputs=[rotation_z_slider, obj_index, type_rotation, z_index]).then(render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4])

                            # gr.HTML("""<div align="center">Scale</div>""")
                            with gr.Group():
                                scale = obj['scale']
                                scale_x_slider = gr.Slider(minimum=0.1, maximum=5, value=scale[0], label="scale_x", step=0.01)
                                scale_y_slider = gr.Slider(minimum=0.1, maximum=5, value=scale[1], label="scale_y", step=0.01)
                                scale_z_slider = gr.Slider(minimum=0.1, maximum=5, value=scale[2], label="scale_z", step=0.01)
                                scale_x_slider.change(updateState, inputs=[scale_x_slider, obj_index, type_scale, x_index]).then(render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4])
                                scale_y_slider.change(updateState, inputs=[scale_y_slider, obj_index, type_scale, y_index]).then(render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4])
                                scale_z_slider.change(updateState, inputs=[scale_z_slider, obj_index, type_scale, z_index]).then(render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4])

                            remove_btn = gr.Button("Remove")
                            remove_btn.click(removeObj, obj_index, object_states)
            
        def isSelected(x):
            print(x)
            print(x!={})
            return x!={}
        # select image
        src_image_path.change(fn=lambda x: gr.update(interactive=False, variant='secondary') if x is None else gr.update(interactive=True, variant='primary'), inputs=src_image_path, outputs=btn_1).then(
                lambda _: gr.Tabs(selected=1), outputs=render_tab    
            )
        btn_1.click(lambda _: gr.Tabs(selected=0), outputs=render_tab)
        # reconstruct image
        btn_1.click(
                fn=reconstruct_image, 
                inputs=[model_states, src_image_path, gen_scale, scene_dict, interactive_state, temp_image, hr_spp], 
                outputs=[res_image, scene_dict, interactive_state, auto_rescale_factor]
            ).then(
                lambda _: gr.Tabs(selected=1), outputs=input_tab
            ).then(
                render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4]
            ).then(
                lambda _: gr.update(interactive=True, variant='primary'), outputs=[btn_2]
            )

        # insert object
        res_image.select(get_pixel_coord, None, [coord2D]).then( # object insertion from clicking image
                get_position_normal, [scene_dict, coord2D], [coords3D, normal3D]
            ).then(
                isSelected, selected_obj_state
            ).then(
                insert_object, [object_states, selected_obj_state, coords3D, auto_rescale_factor], outputs=[object_states]
            ).then(
                render, [scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], [res_image, btn_4]
            )
        # update preview rendering
        btn_2.click(fn=render, inputs=[scene_dict, interactive_state, gr.State(1), preview_spp, gr.State(0), object_states], outputs=[res_image, btn_4])

    gui.launch(share=True)

if __name__ == '__main__':
    main()