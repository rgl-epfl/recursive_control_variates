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
from experiments import scene_kitchen
from plugins.welford import StatisticsEstimator, WelfordVarianceEstimator

output_dir = os.path.join(code_dir, "output", os.path.basename(os.path.dirname(__file__)))
os.makedirs(output_dir, exist_ok=True)

opt_config = scene_kitchen(output_dir, "baseline")

n_steps = 250
spp = 1
integrator = mi.load_dict({'type': 'path', 'max_depth': 64})
scene = opt_config['scene']

params = mi.traverse(scene)

methods = ["analytic", "baseline", "recursive"]
ref_path = os.path.join(output_dir, "img_ref.exr")
if os.path.exists(ref_path):
    img_ref = mi.Bitmap(ref_path)
else:
    img_ref = mi.TensorXf(0.0)
    n_passes = 16
    for i in trange(n_passes):
        img_ref += mi.render(scene, spp=4096, integrator=integrator, seed=i) / n_passes
    mi.Bitmap(img_ref).write(os.path.join(output_dir, "img_ref.exr"))

for i, method in enumerate(methods):
    img = 0.0
    F = 0.0
    H = 0.0
    v_n = 0.0
    var_n = WelfordVarianceEstimator()
    stats = StatisticsEstimator()
    weight_dir = os.path.join(output_dir, method, "weights")
    img_dir = os.path.join(output_dir, method, "img")
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    var = np.zeros(n_steps)
    for j in trange(n_steps):
        F = mi.render(scene, integrator=integrator, seed=j, spp=spp)
        # No need for a second rendering here, as the scene does not evolve
        H = F
        if j > 0:
            if method == "analytic":
                w_s = dr.full(mi.TensorXf, j / (j + 1), F.shape)
            else:
                if j > 5:
                    v_0, v_1, cov = stats.get()
                    if method == "baseline":
                        v_n = var_n.get()
                    else:
                        v_n = w_s ** 2 * (v_n + v_0) + v_1 - 2*w_s * cov

                    w_s = cov / (v_0 + v_n)
                    dr.schedule(w_s, v_n)

                    w_s = dr.select(dr.isnan(w_s) | dr.isinf(w_s), 0.0, w_s)
                    w_s = dr.clamp(w_s, 0.0, 1.0)
                else:
                    w_s = mi.TensorXf(0.0, F.shape)

                stats.update(H, F)

            img = w_s * (img - H) + F
            var_n.update(img)
            mi.Bitmap(w_s).write(os.path.join(weight_dir, f"{j:03d}.exr"))
        else:
            img = F
        var[j] = dr.mean(dr.sqr(img - img_ref))[0]
        mi.Bitmap(img).write(os.path.join(img_dir, f"{j:03d}.exr"))
    np.save(os.path.join(output_dir, method, "var.npy"), var)
