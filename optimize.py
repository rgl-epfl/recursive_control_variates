import mitsuba as mi
import drjit as dr
import numpy as np
from tqdm import tqdm
import os

from utils import *
from vgg import VGGLoss

def run_opt(params):

    scene = params['scene']
    scene_ref = params.get('scene_ref', scene)

    loss_name = params.get('loss', 'L1')
    if loss_name == "L1":
        loss_func = l1_loss
    elif loss_name == "L2":
        loss_func = l2_loss
    elif loss_name == "VGG":
        loss_func = VGGLoss()
    else:
        raise ValueError("Unknown loss function")

    method = params.get('method', 'baseline')

    output_dir = params['output_dir']

    n_steps = params.get('n_steps', 0)
    lr = params.get('lr', 1e-2)
    spp = params.get('spp', 1) # SPP for the forward pass
    spp_grad = params.get('spp_grad', spp) # SPP for gradient estimation
    spp_ref = params.get('spp_ref', 1024) # SPP for the reference image
    benchmark = params.get('benchmark', False)

    assert len(scene.sensors()) == 1, "Only scenes with a single sensor are supported. Consider using the batch sensor to use several viewpoints."

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    scene_params = mi.traverse(scene)

    recomp_freq = params.get('recomp_freq', 10)
    result_dict = {
        'loss': np.zeros(1 + n_steps // recomp_freq, dtype=np.float32),
        'var': np.zeros(1 + n_steps // recomp_freq, dtype=np.float32),
        'runtime': np.zeros((n_steps, 2), dtype=np.float32)
    }

    if 'integrator' in params:
        base_integrator = params['integrator']
    else:
        raise ValueError("No integrator specified")

    integrator_dict = {
        'type': 'meta',
        'method': method,
        'denoise': params.get('denoise', False),
        'integrator': base_integrator,
        'adjoint_integrator': params.get('adjoint_integrator', base_integrator),
        'beta1': params.get('beta1', 0.9),
        'beta2': params.get('beta2', 0.999),
        'force_baseline': params.get('force_baseline', False),
        'pre_update': params.get('pre_update', False)
    }
    integrator = mi.load_dict(integrator_dict)

    # Texture/volume upsampling frequency
    upsample_steps = []
    for w in params.get('upsample', []):
        assert w > 0 and w < 1
        upsample_steps.append(int(n_steps * w))

    # Step size schedule
    schedule_steps = []
    for w in params.get('schedule', []):
        assert w > 0 and w < 1
        schedule_steps.append(int(n_steps * w))

    # Render the reference image
    img_ref = render_reference(params, scene_ref, base_integrator)

    # Render the reference at the display resolution
    ref_display_path = os.path.join(os.path.dirname(output_dir), "img_ref_display.exr")
    if 'final_res' in params and not os.path.exists(ref_display_path):
        img_display = render_display(params, scene_ref, scene_params, base_integrator)
        mi.Bitmap(img_display).write(ref_display_path)

    # Initialize the optimizer
    opt = mi.ad.Adam(lr=lr)

    # Initialize the parameters
    initialize_parameters(params, opt, scene, scene_params)

    # Render the starting point
    save_path = os.path.join(os.path.dirname(params['output_dir']), f"img_start.exr")
    if not os.path.isfile(save_path):
        with dr.suspend_grad():
            start_img = mi.render(scene, seed = 2048, integrator=base_integrator, spp=spp_ref)
            mi.Bitmap(start_img).write_async(save_path)

    # Main loop
    with tqdm(total=n_steps, bar_format="{l_bar}{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]", desc=f"{os.path.basename(os.path.dirname(params['output_dir'])):<20} | {method:<20}") as pbar:
        for i in range(n_steps):
            active_benchmark = benchmark and i > 16
            with dr.scoped_set_flag(dr.JitFlag.KernelHistory, active_benchmark):
                scene_params.update(opt)
                img = mi.render(scene, scene_params, integrator=integrator, seed=i, spp=spp, spp_grad=spp_grad)

                loss = loss_func(img, img_ref)
                if loss_name == 'VGG':
                    loss += l1_loss(img, img_ref)

                if active_benchmark:
                    dr.schedule(loss)
                    result_dict['runtime'][i, 0] = runtime()


            checkpoint(scene, params, i, integrator, loss_func, img, img_ref, result_dict)

            with dr.scoped_set_flag(dr.JitFlag.KernelHistory, active_benchmark):
                dr.backward(loss)

                # Apply gradient preconditioning if needed
                precondition(params, opt)
                # Gradient descent
                opt.step()
                # Clamp parameters if needed
                clamp(params, opt)

                if active_benchmark:
                    result_dict['runtime'][i, 1] = runtime()

            # Volume upsampling
            if i in upsample_steps:
                upsample(scene, params, opt, integrator)

            # Update step size
            if i in schedule_steps:
                lr *= 0.5
                opt.set_learning_rate(lr)

            pbar.update(1)

    with dr.suspend_grad():
        scene_params.update(opt)
        # Get the final state
        img_final = render_display(params, scene_ref, scene_params, base_integrator)
        mi.Bitmap(img_final).write(os.path.join(output_dir, "img_final.exr"))

    # Save the final state
    save_final_state(params, opt, output_dir)

    return result_dict
