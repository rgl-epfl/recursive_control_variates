import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
import os
from mitsuba.scalar_rgb import Transform4f as T
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
sys.path.append(Path(__file__).parents[3])
from experiments import SCENES_DIR, OUTPUT_DIR

from tqdm import trange

target = [0.5, 0.5, 0.5]
d = 1
h = 0.5
sensor_count = 8
i = 0
theta = 2*np.pi / sensor_count * i
sensor = {
    'type': 'perspective',
    'fov': 72.734,
    'to_world': T.look_at(target=target, origin=[target[0]+d*np.cos(theta), h, target[2]+d*np.sin(theta)], up=[0, 1, 0]),
    'film': {
        'type': 'hdrfilm',
        'width': 1280, 'height': 1280,
    },
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
        'interior': medium
    },
    'envmap': {
        'type': 'envmap',
        'filename': os.path.join(SCENES_DIR, 'common/textures/kloofendal_38d_partly_cloudy_4k.exr'),
        'scale': 1.0,
    },
    'sensor': sensor
}

scene_name = 'dust_devil'
scene = mi.load_dict(scene_dict)
integrator = mi.load_dict({'type': 'prbvolpath', 'max_depth':64, 'rr_depth': 64})
img_ref = mi.TensorXf(0.0)

n_passes = 32
spp = 128
for i in trange(n_passes):
    img_ref += mi.render(scene, integrator=integrator, seed=i, spp=spp) / n_passes

mi.Bitmap(img_ref).write(os.path.join(OUTPUT_DIR, scene_name, 'img_ref_re.exr'))

dirs = [os.path.join(OUTPUT_DIR, scene_name, 'baseline'),
        os.path.join(OUTPUT_DIR, f"{scene_name}_high_spp", 'baseline'),
        os.path.join(OUTPUT_DIR, scene_name, 'cv_ps'),]

keys = ['object.interior_medium.albedo.data', 'object.interior_medium.sigma_t.data']
for i, d in enumerate(dirs):
    scene_params = mi.traverse(scene)
    for key in keys:
        final_volume = mi.load_dict({
            'type': 'gridvolume',
            'filename': os.path.join(d, f"{key.replace('.', '_')}_final.vol"),
        })
        scene_params[key] = mi.traverse(final_volume)['data']
    scene_params.update()

    img = mi.TensorXf(0.0)
    for i in trange(n_passes):
        img += mi.render(scene, integrator=integrator, seed=i, spp=spp) / n_passes

    mi.Bitmap(img).write(os.path.join(d, 'img_final_re.exr'))
