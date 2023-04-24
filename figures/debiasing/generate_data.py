import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
from tqdm import trange

import numpy as np

import os
import sys
from pathlib import Path
import argparse

code_dir = str(Path(__file__).parents[2])
sys.path.append(os.path.dirname(code_dir))
from experiments import scene_dragon

output_dir = os.path.join(code_dir, "output", os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

opt_config = scene_dragon(output_dir, "baseline")

n_steps = 250
n_runs = 64
spp = opt_config['spp']
spp_grad = opt_config['spp_grad']
spp_ref = opt_config['spp_ref']
integrator = opt_config['integrator']
scene = opt_config['scene']

seed_offset = n_steps
params = mi.traverse(scene)

img_ref = mi.render(scene, params, integrator=integrator, spp=spp_ref)
mi.Bitmap(img_ref).write(os.path.join(output_dir, "img_ref.exr"))

def save_image(img, output_dir, img_name, it):
    filename = os.path.join(output_dir, img_name, f"{it:04d}.exr")
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    mi.Bitmap(img).write_async(filename)

for j, debias in enumerate([True, False]):
    loss_hist = np.zeros(n_steps)
    save_dir = os.path.join(output_dir, "debiased" if debias else "baseline")
    opt = mi.ad.Adam(lr=5e-2)

    # Initialize the parameters
    for key, param in opt_config['variables'].items():
        init_state = param['init']
        if 'sigma_t' in key:
            opt[key] = mi.Float(init_state)
            params[key] = opt[key]
        else:
            params[key] = init_state
			# Add the parameter to the optimizer
            opt[key] = params[key]

    for i in trange(n_steps):
        params.update(opt)

        # Primal rendering
        img = mi.render(scene, params, integrator=integrator, seed=i, spp=spp, spp_grad=spp_grad)
        grad_biased = dr.sign(img - img_ref)

        if debias:
            # Compute bias term
            with dr.suspend_grad():
                img_inf = mi.render(scene, params, integrator=integrator, seed=1457892+i, spp=spp_ref)
                grad_inf = dr.sign(img_inf - img_ref)

                grad_exp = 0
                for run in range(n_runs):
                    x = mi.render(scene,  params, seed=n_steps+i*n_runs+run, spp=spp)
                    grad_exp += dr.sign(x - img_ref) / n_runs

                grad = grad_biased + grad_inf - grad_exp
        else:
            grad = dr.detach(grad_biased)

        dr.backward(img * grad / dr.prod(img.shape))

        with dr.suspend_grad():
            if not debias:
                img_inf = mi.render(scene, params, integrator=integrator, seed=i+1, spp=spp_ref)

            save_image(img_inf, save_dir, "img_inf", i)
            save_image(img, save_dir, "img", i)

            loss_hist[i] = dr.mean(dr.abs(img_inf - img_ref))[0]

        opt.step()

        for key, param in opt_config['variables'].items():
            opt[key] = dr.clamp(dr.detach(opt[key]), param['clamp'][0], param['clamp'][1])

    np.savez(os.path.join(save_dir, "loss_hist.npz"), loss_hist=loss_hist)
