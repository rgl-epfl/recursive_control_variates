import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr

import numpy as np

import os
import sys
import argparse

import plugins
from optimize import run_opt
from largesteps import *

from mitsuba.scalar_rgb import Transform4f as T

SCENES_DIR = os.path.join(os.path.dirname(__file__), "scenes")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

def scene_janga(method, output_dir):
    sensor_count = 8
    # Visualisation resolution
    final_res = (720*sensor_count, 720)
    # Training resolution
    resx = 256
    resy = 256

    batch_sensor = {
        'type': 'batch',
        'film': {
            'type': 'hdrfilm',
            'width': resx*sensor_count, 'height': resy,
        },
        'sampler': {
            'type': 'independent',
            'sample_count': 1
        }
    }

    scale = [1.0, 1.2, 1.5]
    target = [0.5 * s for s in scale]
    d = 2
    h = 0
    for i in range(sensor_count):
        theta = 2*np.pi / sensor_count * i + 0.1
        batch_sensor[f"sensor_{i:02d}"] = {
            'type': 'perspective',
            'fov': 45,
            'to_world': T.look_at(target=target, origin=[target[0]+d*np.cos(theta), h, target[2]+d*np.sin(theta)], up=[0, 1, 0]),
        }

    medium = {
        'type': 'heterogeneous',
        'sigma_t': {
            'type': 'gridvolume',
            'filename': os.path.join(SCENES_DIR, 'janga-smoke/volumes/janga-smoke-264-136-136.vol'),
            'to_world': T.scale(scale),
            'use_grid_bbox': False,
            'accel': False
        },
        'albedo': {
            'type': 'gridvolume',
            'filename': os.path.join(SCENES_DIR, 'janga-smoke/volumes/albedo-noise-256-128-128.vol'),
            'to_world': T.scale(scale),
        },
        'scale': 20.0,

        'sample_emitters': True,
        'has_spectral_extinction': False,
        'majorant_resolution_factor': 0
    }

    scene_dict = {
        'type': 'scene',
        'object': {
            'type': 'obj',
            'filename': os.path.join(SCENES_DIR, 'common/meshes/cube_unit.obj'),
            'bsdf': {'type': 'null'},
            'to_world': T.scale(scale),
        },
        'envmap': {
            'type': 'envmap',
            'filename': os.path.join(SCENES_DIR, 'common/textures/alps_field_4k.exr'),
            'scale': 0.5,
        },
        'sensor': batch_sensor
    }

    if method == 'cv_ps':
        scene_dict['object']['interior'] = {
            'type': 'twostatemedium',
            'old': medium,
            'new': medium,
            'incoming': medium
        }
    else:
        scene_dict['object']['interior'] = medium

    v_res = 16
    scene = mi.load_dict(scene_dict)
    params = {
        'scene': scene,
        'variables': {
            'object.interior_medium.sigma_t.data':  {
                'init': dr.full(mi.TensorXf, 0.04, (v_res, v_res, v_res, 1)),
                'clamp': (0.0, 250.0),
            },
            'object.interior_medium.albedo.data':  {
                'init': dr.full(mi.TensorXf, 0.6, (v_res, v_res, v_res, 3)),
                'clamp': (0.0, 1.0),
                'lr_factor': 2.0
            },
        },
        'use_majorant_supergrid': True,
        'recomp_freq': 50,
        'upsample': [0.04, 0.16, 0.36, 0.64],
        'schedule': [0.75, 0.85, 0.95],
        'save': True,
        'benchmark': True,
        'n_steps': 2000,
        'method': method,
        'lr': 5e-3,
        'spp': 4,
        'spp_grad': 4,
        'spp_ref': 128,
        'ref_passes': 16,
        'spp_inf': 64,
        'final_res': final_res,
        'output_dir': output_dir,
        'integrator': mi.load_dict({'type': 'twostateprbvolpath' if method == 'cv_ps' else 'prbvolpath', 'max_depth': 64, 'rr_depth': 64}),
        'adjoint_integrator': mi.load_dict({'type': 'volpathsimple', 'max_depth': 64, 'rr_depth': 64})
    }
    return params

def scene_dust_devil(method, output_dir):
    v_res = 16
    sensor_count = 8
    final_res = (720*sensor_count, 1280)
    resx = 256
    resy = 456

    batch_sensor = {
        'type': 'batch',
        'film': {
            'type': 'hdrfilm',
            'width': resx*sensor_count, 'height': resy,
        },
        'sampler': {
            'type': 'independent',
            'sample_count': 1
        }
    }

    target = [0.5, 0.5, 0.5]
    d = 1
    h = 0.5
    for i in range(sensor_count):
        theta = 2*np.pi / sensor_count * i
        batch_sensor[f"sensor_{i:02d}"] = {
            'type': 'perspective',
            'fov': 45,
            'to_world': T.look_at(target=target, origin=[target[0]+d*np.cos(theta), h, target[2]+d*np.sin(theta)], up=[0, 1, 0]),
        }

    medium = {
        'type': 'heterogeneous',
        'sigma_t': {
            'type': 'gridvolume',
            'filename': os.path.join(SCENES_DIR, 'dust-devil/volumes/embergen_dust_devil_tornado_a_50-256-256-256.vol'),
            'use_grid_bbox': False,
            'accel': False
        },
        'albedo': {
            'type': 'gridvolume',
            'filename': os.path.join(SCENES_DIR, 'dust-devil/volumes/albedo-constant-sand-256-256-256.vol'),
            'use_grid_bbox': False,
            'accel': False
        },
        'phase': {
            'type': 'hg',
            'g': 0.877
        },
        'scale': 100.0,

        'sample_emitters': True,
        'has_spectral_extinction': False,
        'majorant_resolution_factor': 0
    }
    scene_dict = {
        'type': 'scene',
        'object': {
            'type': 'obj',
            'filename': os.path.join(SCENES_DIR, 'common/meshes/cube_unit.obj'),
            'bsdf': {'type': 'null'},
        },
        'envmap': {
            'type': 'envmap',
            'filename': os.path.join(SCENES_DIR, 'common/textures/kloofendal_38d_partly_cloudy_4k.exr'),
            'scale': 1.0,
        },
        'sensor': batch_sensor
    }

    if method == 'cv_ps':
        scene_dict['object']['interior'] = {
            'type': 'twostatemedium',
            'old': medium,
            'new': medium,
            'incoming': medium
        }
    else:
        scene_dict['object']['interior'] = medium

    scene = mi.load_dict(scene_dict)
    params = {
        'scene': scene,
        'variables': {
            'object.interior_medium.sigma_t.data':  {
                'init': dr.full(mi.TensorXf, 0.04, (v_res, v_res, v_res, 1)),
                'clamp': (0.0, 250.0),
            },
            'object.interior_medium.albedo.data':  {
                'init': dr.full(mi.TensorXf, 0.6, (v_res, v_res, v_res, 3)),
                'clamp': (0.0, 1.0),
                'lr_factor': 2.0
            },
        },
        'use_majorant_supergrid': True,
        'recomp_freq': 50,
        'upsample': [0.04, 0.16, 0.36, 0.64],
        'schedule': [0.75, 0.85, 0.95],
        'save': True,
        'benchmark': True,
        'n_steps': 2000,
        'method': method,
        'lr': 5e-4,
        'spp': 16,
        'spp_grad': 16,
        'spp_ref': 128,
        'ref_passes': 16,
        'spp_inf': 64,
        'final_res': final_res,
        'output_dir': output_dir,
        'integrator': mi.load_dict({'type': 'twostateprbvolpath' if method == 'cv_ps' else 'prbvolpath', 'max_depth': 64, 'rr_depth': 64}),
        'adjoint_integrator': mi.load_dict({'type': 'volpathsimple', 'max_depth': 64, 'rr_depth': 64})

    }

    return params

def scene_rover(method, output_dir):
    sensor_count = 10
    final_res = (720*sensor_count), 720
    resx = 256
    resy = 256
    v_res = 16

    batch_sensor = {
        'type': 'batch',
        'film': {
            'type': 'hdrfilm',
            'width': resx*sensor_count, 'height': resy,
        },
        'sampler': {
            'type': 'independent',
            'sample_count': 1
        }
    }

    scale = [6.5, 7.5, 10]
    tr = [-3, -1, -5]
    target = [-0.25, 2.75, 0]
    d = 12
    h = 10.0
    for i in range(sensor_count):
        theta = 2*np.pi / sensor_count * i + 0.1
        batch_sensor[f"sensor_{i:02d}"] = {
            'type': 'perspective',
            'fov': 45,
            'to_world': T.look_at(target=target, origin=[target[0]+d*np.cos(theta), h, target[2]+d*np.sin(theta)], up=[0, 1, 0]),
        }

	# Render scene ref here

    to_world = T.translate(tr).scale(scale)
    medium = {
        'type': 'heterogeneous',
        'sigma_t': {
            'type': 'gridvolume',
            'filename': os.path.join(SCENES_DIR, 'common/volumes/sigma_t-constant-sand-256-256-256.vol'),
            'to_world': to_world,
            'use_grid_bbox': False,
            'accel': False
        },
        'albedo': {
            'type': 'gridvolume',
            'filename': os.path.join(SCENES_DIR, 'common/volumes/albedo-constant-sand-256-256-256.vol'),
            'to_world': to_world,
            'use_grid_bbox': False,
            'accel': False
        },
        'scale': 1.0,

        'sample_emitters': True,
        'has_spectral_extinction': False,
        'majorant_resolution_factor': 8
    }

    scene_dict = {
        'type': 'scene',
        'object': {
            'type': 'obj',
            'filename': os.path.join(SCENES_DIR, 'common/meshes/cube_unit.obj'),
            'bsdf': {'type': 'null'},
            'to_world': to_world,
        },
        'envmap': {
            'type': 'envmap',
            'filename': os.path.join(SCENES_DIR, 'common/textures/clarens_night_02_4k.exr'),
        },
        'sensor': batch_sensor
    }

    if method == 'cv_ps':
        scene_dict['object']['interior'] = {
            'type': 'twostatemedium',
            'old': medium,
            'new': medium,
            'incoming': medium
        }
    else:
        scene_dict['object']['interior'] = medium


    params = {
        'scene': mi.load_dict(scene_dict),
        'variables': {
            'object.interior_medium.sigma_t.data':  {
                'init': dr.full(mi.TensorXf, 0.04, (v_res, v_res, v_res, 1)),
                'clamp': (0.0, 250.0),
            },
            'object.interior_medium.albedo.data':  {
                'init': dr.full(mi.TensorXf, 0.6, (v_res, v_res, v_res, 3)),
                'clamp': (0.0, 1.0),
                'lr_factor': 2.0
            },
        },
        'upsample': [0.04, 0.16, 0.36, 0.64],
        'schedule': [0.75, 0.85, 0.95],
        'save': True,
        'benchmark': True,
        'n_steps': 2000,
        'recomp_freq': 50,
        'method': method,
        'lr': 1e-2,
        'spp': 8,
        'spp_grad': 8,
        'spp_ref': 128,
        'ref_passes': 16,
        'spp_inf': 64,
        'final_res': final_res,
        'output_dir': output_dir,
        'integrator': mi.load_dict({'type': 'twostateprbvolpath' if method == 'cv_ps' else 'prbvolpath', 'max_depth': 64, 'rr_depth': 64}),
        'adjoint_integrator': mi.load_dict({'type': 'volpathsimple', 'max_depth': 64, 'rr_depth': 64})

    }

    ref_name = os.path.join(os.path.dirname(output_dir), "img_ref.exr")
    if not os.path.exists(ref_name):
        sensor = mi.load_dict(batch_sensor)
        scene = mi.load_file(os.path.join(SCENES_DIR, "rover", "rover-ref.xml"), envmap_filename=os.path.join(SCENES_DIR, scene_dict['envmap']['filename']))
        img_ref = mi.TensorXf(0.0)
        ref_passes = params['ref_passes']
        spp_ref = params['spp_ref']
        from tqdm import trange
        for j in trange(ref_passes):
            img_ref += mi.render(scene, seed = 17843 + j, sensor=sensor, integrator=params['integrator'], spp=spp_ref) / ref_passes
        mi.Bitmap(img_ref).write_async(ref_name)
        sensor_params = mi.traverse(sensor)

        sensor_params['film.size'] = final_res
        sensor_params.update()
        img = mi.TensorXf(0.0)
        for j in trange(ref_passes):
            img += mi.render(scene, seed = 17843 + j, sensor=sensor, integrator=params['integrator'], spp=spp_ref) / ref_passes
        mi.Bitmap(img).write_async(os.path.join(os.path.dirname(output_dir), "img_ref_display.exr"))

        del scene
        del sensor

    return params

def scene_bunnies(method, output_dir):
    scene = mi.load_file(os.path.join(SCENES_DIR, "bunnies", "scene_twostates.xml" if method == 'cv_ps' else "scene.xml"), resx=1280, resy=720)
    params = {
        'scene': scene,
        'variables': {
            'PLYMesh_1.interior_medium.sigma_t.value.value': {
                'init': 1.0,
                'clamp': (1e-4, 10.0),
            },
            'PLYMesh_2.bsdf.alpha.value': {
                'init': 0.5,
                'clamp': (1e-4, 1.0),
            },
            'PLYMesh_3.bsdf.reflectance.value': {
                'init': [0.5, 0.5, 0.5],
                'clamp': (1e-4, 1.0),
            }
        },
        'save': True,
        'n_steps': 300,
        'recomp_freq': 1,
        'method': method,
        'lr': 5e-2,
        'spp': 1,
        'spp_grad': 16,
        'spp_ref': 4096,
        'spp_inf': 1024,
        'output_dir': output_dir,
        'integrator': mi.load_dict({'type': 'twostateprbvolpath' if method == 'cv_ps' else 'prbvolpath', 'max_depth': 64, 'rr_depth': 64}),
    }

    return params

def scene_ajar(method, output_dir):
    scene = mi.load_file(os.path.join(SCENES_DIR, "veach-ajar", "scene_twostates.xml" if method == 'cv_ps' else "scene.xml"), resx=1280, resy=720)
    lambda_ = 29.0
    return {
        'scene': scene,
        'variables': {
            'LandscapeBSDF.reflectance.data': {
               'init': dr.full(mi.TensorXf, 0.1, (256, 512, 3)),
                'clamp': (1e-4, 1.0),
                'largesteps': CholeskySolver(dr.full(mi.TensorXf, 0.1, (256, 512, 3)), lambda_)
            },
            'tea.albedo.value.value': {
                'init': [0.1, 0.1, 0.1],
                'clamp': (1e-4, 1.0)
            },
            'tea.sigma_t.value.value': {
                'init': 0.1,
                'clamp': (1e-4, 10.0),
                'lr_factor': 2.0
            }
        },
        'save': True,
        'schedule': [0.75, 0.85, 0.95],
        'n_steps': 500,
        'recomp_freq': 5,
        'method': method,
        'denoise': False,
        'lr': 2e-2,
        'spp': 16,
        'spp_grad': 16,
        'spp_ref': 4096,
        'ref_passes': 16,
        'spp_inf': 4096,
        'output_dir': output_dir,
        'integrator': mi.load_dict({'type': 'twostateprbvolpath' if method == 'cv_ps' else 'prbvolpath', 'max_depth': 64})
    }

def scene_ajar_bias(method, output_dir):
    params = scene_ajar(method, output_dir)
    params['bias_steps'] = [255]
    return params

def scene_dragon(method, output_dir):
    scene = mi.load_file(os.path.join(SCENES_DIR, "dragon", "scene_twostates.xml" if method == 'cv_ps' else "scene.xml"), resx=500, resy=300)
    params = {
        'scene': scene,
        'variables': {
            'PLYMesh_2.interior_medium.sigma_t.value.value': {
                'init': 0.1,
                'clamp': (1e-4, 10.0),
            }
        },
        'save': True,
        'n_steps': 750,
        'method': method,
        'lr': 5e-2,
        'spp': 1,
        'spp_grad': 16,
        'spp_ref': 2048,
        'spp_inf': 256,
        'output_dir': output_dir,
        'integrator': mi.load_dict({'type': 'twostateprbvolpath' if method == 'cv_ps' else 'prbvolpath', 'max_depth': 64, 'rr_depth': 64}),
    }

    return params

def scene_teapot(method, output_dir):
    scene = mi.load_file(os.path.join(SCENES_DIR, "teapot-full", "scene_twostates" if method == 'cv_ps' else "scene.xml"), resx=500, resy=300)
    params = {
        'scene': scene,
        'variables': {
            'tea.sigma_t.value.value': {
                'init': 0.1,
                'clamp': (1e-4, 1.0),
            },
            'tea.albedo.value.value': {
                'init': [0.2, 0.2, 0.2],
                'clamp': (1e-4, 1.0),
            }
        },
        'save': True,
        'n_steps': 250,
        'method': method,
        'lr': 1e-2,
        'spp': 1,
        'spp_grad': 16,
        'spp_ref': 8192,
        'spp_inf': 64,
        'output_dir': output_dir,
        'integrator': mi.load_dict({'type': 'twostateprbvolpath' if method == 'cv_ps' else 'prbvolpath', 'max_depth': 64, 'rr_depth': 64})
    }

    return params

def scene_cornell_vgg(method, output_dir):
    scene_dict = mi.cornell_box()
    scene_dict['sensor']['film']['width'] = 512
    scene_dict['sensor']['film']['height'] = 512
    back_bsdf = {
        'type': 'diffuse',
        'reflectance': {
            'type': 'bitmap',
            'filename': os.path.join(SCENES_DIR, 'concrete.exr'),
            'to_uv': mi.ScalarTransform4f.rotate([1,0,0], 180)
        }
    }

    if method == 'cv_ps':
        scene_dict['back']['bsdf'] = {
            'type': 'twostate',
            'old': back_bsdf,
            'new': back_bsdf,
            'incoming': back_bsdf
        }
    else:
        scene_dict['back']['bsdf'] = back_bsdf

    params = {
        'scene': mi.load_dict(scene_dict),
        'variables': {
            'back.bsdf.reflectance.data': {
                'init': dr.full(mi.TensorXf, 0.5, (512, 512, 3)),
                'clamp': (1e-3, 1.0),
            }
        },
        'save': True,
        'n_steps': 500,
        'final_res': (1280, 1280),
        'loss': 'VGG',
        'method': method,
        'lr': 2e-2,
        'spp': 1,
        'spp_grad': 16,
        'ref_passes': 8,
        'spp_ref': 1024,
        'spp_inf': 128,
        'output_dir': output_dir,
        'integrator': mi.load_dict({'type': 'twostateprb' if method == 'cv_ps' else 'prb', 'max_depth': 64, 'rr_depth': 64}),
    }
    return params

def scene_kitchen(method, output_dir):
    scene = mi.load_file(os.path.join(SCENES_DIR, "kitchen", "scene.xml"))
    return {
        'scene': scene,
        'variables': {},
        'save': True,
        'n_steps': 1,
        'method': method,
        'denoise': False,
        'lr': 2e-2,
        'spp': 1,
        'spp_grad': 16,
        'spp_ref': 4096,
        'ref_passes': 4,
        'output_dir': output_dir,
        'integrator': mi.load_dict({'type': 'twostateprb' if method == 'cv_ps' else 'prb', 'max_depth': 64, 'rr_depth': 64}),
    }

AVAILABLE_SCENES = [name[6:] for name in globals() if name.startswith('scene_')]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a Mitsuba experiment')
    parser.add_argument('scene', type=str, choices=AVAILABLE_SCENES, help='Name of the scene to use')
    parser.add_argument('method', type=str, help='Name of the optimization method to use')
    # Optional overrides of scene parameters
    parser.add_argument('--n_steps', type=int, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--spp', type=int, help='Samples per pixel for the primal rendering')
    parser.add_argument('--spp_grad', type=int, help='Samples per pixel for the adjoint rendering')
    parser.add_argument('--beta1', type=float, help='β₁ parameter for statistics')
    parser.add_argument('--beta2', type=float, help='β₂ parameter for statistics')

    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(__file__), "output", args.scene, args.method)
    params = globals()[f"scene_{args.scene}"](args.method, output_dir)

    # Override the parameters if necessary
    for arg in ['n_steps', 'lr', 'spp', 'spp_grad', 'beta1', 'beta2']:
        if getattr(args, arg) is not None:
            params[arg] = getattr(args, arg)

    result_dict = run_opt(params)

    np.savez(os.path.join(output_dir, "result.npz"), **result_dict)
