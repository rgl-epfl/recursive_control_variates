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

import plugins
from experiments import scene_dragon
from utils import l1_loss, l2_loss

output_dir = os.path.join(code_dir, "output", os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# opt_config = scene_teapot(output_dir, "baseline")
opt_config = scene_dragon(output_dir, "baseline")

spp = opt_config['spp']
spp_grad = opt_config['spp_grad']
spp_ref = opt_config['spp_ref']
integrator = opt_config['integrator']
scene = opt_config['scene']

params = mi.traverse(scene)
key = 'PLYMesh_2.interior_medium.sigma_t.value.value'

img_ref = mi.render(scene, params, integrator=integrator, spp=spp_ref)
mi.Bitmap(img_ref).write(os.path.join(output_dir, "img_ref.exr"))

spps = 2 ** (2*np.arange(5) + 1)
losses = ["l1", "l2"]
n_runs = 2 ** np.arange(8, 0, -1)

param_landscape = np.linspace(2, 8, 25)

np.save(os.path.join(output_dir, "spps.npy"), spps)
np.save(os.path.join(output_dir, "param_landscape.npy"), param_landscape)

for loss_name in losses:
    for k, spp in enumerate(spps):
        loss_hist = np.zeros(len(param_landscape))
        grad_hist = np.zeros_like(loss_hist)
        save_dir = os.path.join(output_dir, loss_name, str(spp))

        print(f"Loss: {loss_name}, SPP: {spp}")
        for j in trange(n_runs[k]):
            for i, param in enumerate(param_landscape):
                # Update the extinction
                params[key] = mi.Float(param)
                dr.set_grad_enabled(params[key], True)
                params.update()

                # Primal rendering
                img = mi.render(scene, params, integrator=integrator, seed=i+j*len(param_landscape), spp=spp, spp_grad=32)

                if loss_name == "l1":
                    loss = l1_loss(img, img_ref)
                elif loss_name == "l2":
                    loss = l2_loss(img, img_ref)
                else:
                    raise NotImplementedError

                dr.backward(loss)

                loss_hist[i] += loss[0] / n_runs[k]
                grad_hist[i] += dr.grad(params[key])[0] / n_runs[k]

        np.save(os.path.join(output_dir, f"loss_{loss_name}_{spp:04d}.npy"), loss_hist)
        np.save(os.path.join(output_dir, f"grad_{loss_name}_{spp:04d}.npy"), grad_hist)
