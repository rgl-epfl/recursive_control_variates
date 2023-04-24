import os
import mitsuba as mi
import drjit as dr
from plugins.volpathsimple import get_single_medium
import json
from tqdm import trange

def save_img(img, output_dir, img_name, it, save_png=False):
    filename = os.path.join(output_dir, img_name, f"{it:04d}.exr")
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    mi.Bitmap(img).write_async(filename)
    if save_png:
        mi.util.convert_to_bitmap(img).write_async(filename.replace(".exr", ".png"))

def runtime():
    dr.eval()
    dr.sync_thread()
    return sum([k['execution_time'] + k['codegen_time'] for k in dr.kernel_history()])

def l2_loss(x, y):
    return dr.mean(dr.sqr(x-y))

def l1_loss(x, y):
    return dr.mean(dr.abs(x-y))

def adjust_majorant_res_factor(scene, density_res):
    res_factor = 8

    if res_factor > 1:
        min_side = dr.min(density_res[:3])
        # For the current density res, find the largest factor that
        # results in a meaningful supergrid resolution.
        while (res_factor > 1) and (min_side // res_factor) < 4:
            res_factor -= 1
    # Otherwise, just disable the supergrid.
    if res_factor <= 1:
        res_factor = 0

    medium = get_single_medium(scene)
    current = medium.majorant_resolution_factor()
    if current != res_factor:
        medium.set_majorant_resolution_factor(res_factor)

def save_final_state(params, opt, output_dir):
    final_state = {}
    for key in params['variables'].keys():
        param = opt[key]
        if isinstance(param, mi.TensorXf):
            shape = dr.shape(opt[key])
            if len(shape) == 4: # volume
                mi.VolumeGrid(opt[key]).write(os.path.join(output_dir, f"{key.replace('.', '_')}_final.vol"))
            elif len(shape) == 3: # Texture
                mi.Bitmap(opt[key]).write(os.path.join(output_dir, f"{key.replace('.', '_')}_final.exr"))
        else:
            final_state[key] = param.numpy().tolist()

    if len(final_state) > 0:
        with open(os.path.join(output_dir, "final_state.json"), 'w') as f:
            json.dump(final_state, f)

def initialize_parameters(params, opt, scene, scene_params):
    for i in range(2 if params['method'] == 'cv_ps' else 1):
        # We do this twice to properly set the initial states of the twostate BSDFs/Media
        for key, param in params['variables'].items():
            init_state = param['init']
            if 'sigma_t' in key:
                # Spectrally varying extinction is not supported
                if isinstance(init_state, float):
                    opt[key] = mi.Float(init_state)
                elif isinstance(init_state, mi.TensorXf):
                    assert init_state.shape[-1] == 1
                    opt[key] = init_state

                    if params.get('use_majorant_supergrid', False):
                        adjust_majorant_res_factor(scene, init_state.shape)

                scene_params[key] = opt[key]
            else:
                scene_params[key] = init_state
                opt[key] = scene_params[key]

            # Adjust learning rate if needed
            if 'lr_factor' in param:
                opt.set_learning_rate({key: params.get('lr', 1e-2) * param['lr_factor']})

        scene_params.update()


def d_l(x, y, name):
    if name == 'L1':
        return dr.sign(x - y)
    elif name == 'L2':
        return 2 * (x - y)
    else:
        raise NotImplementedError(f"Unknown loss function {name}")

def render_reference(params, scene, integrator):
    save_path = os.path.join(os.path.dirname(params['output_dir']), f"img_ref.exr")
    ref_passes = params.get('ref_passes', 1)
    spp_ref = params.get('spp_ref', params.get('spp', 1))
    if not os.path.isfile(save_path):
        # Only recompute the reference if it's not already saved, since it's quite expensive
        img_ref = mi.TensorXf(0.0)
        for j in trange(ref_passes):
            img_ref += mi.render(scene, seed = 17843 + j, integrator=integrator, spp=spp_ref) / ref_passes
        mi.Bitmap(img_ref).write_async(save_path)
    else:
        img_ref = mi.TensorXf(mi.Bitmap(save_path))

    return img_ref

def render_display(params, scene, scene_params, integrator):
    opt_res = scene.sensors()[0].film().size()
    final_res = params.get('final_res', opt_res)

    scene_params['sensor.film.size'] = final_res
    scene_params.update()
    ref_passes = params.get('ref_passes', 1)
    spp_ref = params.get('spp_ref', params.get('spp', 1))
    img = mi.TensorXf(0.0)
    for j in trange(ref_passes):
        img += mi.render(scene, seed = 17843 + j, integrator=integrator, spp=spp_ref) / ref_passes

    scene_params['sensor.film.size'] = opt_res
    scene_params.update()
    return img

def checkpoint(scene, params, i, integrator, loss_func, img, img_ref, result_dict):
    recomp_freq = params.get('recomp_freq', 10)
    save = params.get('save', False)
    spp_inf = params.get('spp_inf', 64)
    output_dir = params['output_dir']
    denoise = params.get('denoise', False)
    method = params.get('method', 'baseline')

    if i % recomp_freq == 0:
        with dr.suspend_grad():
            # Re render the current state with a higher sample count, to avoid bias in the loss evaluation
            img_inf = mi.render(scene, integrator=integrator.integrator, seed=i+1, spp=spp_inf)
            if save:
                save_img(img_inf, output_dir, "img_inf", i//recomp_freq, save_png=True)

            result_dict["loss"][i//recomp_freq] = loss_func(img_ref, img_inf).numpy()[0]
            # Compute variance as MSE between the noisy and the high spp rendering
            result_dict["var"][i//recomp_freq] =  dr.mean(dr.sqr(img_inf - img)).numpy()[0]

        if save:
            save_img(integrator.img, output_dir, "img", i//recomp_freq, save_png=True)
            if denoise:
                save_img(img, output_dir, "img_denoised", i//recomp_freq, save_png=True)
            if method != "baseline" and dr.all(integrator.stats.n > integrator.warmup):
                save_img(integrator.w_s, output_dir, "weights", i//recomp_freq)
                save_img(integrator.H, output_dir, "img_H", i//recomp_freq)
                save_img(integrator.F, output_dir, "img_F", i//recomp_freq)

    if 'bias_steps' in params and i in params['bias_steps']:
        with dr.suspend_grad():
            ref_passes = params.get('ref_passes', 1)
            spp_ref = params.get('spp_ref', params.get('spp', 1))
            img_gt = mi.TensorXf(0.0)
            for j in trange(ref_passes):
                img_gt += mi.render(scene, seed = 17843 + j, integrator=integrator, spp=spp_ref) / ref_passes
            save_img(img_gt, output_dir, "bias_gt", i)
            save_img(img, output_dir, "bias_img", i)

def precondition(params, opt):
    for key, param in params['variables'].items():
        if 'largesteps' in param:
            dr.set_grad(opt[key], param['largesteps'].precondition(dr.grad(opt[key])))

def clamp(params, opt):
    for key, param in params['variables'].items():
        if 'clamp' in param:
            opt[key] = dr.clamp(dr.detach(opt[key]), param['clamp'][0], param['clamp'][1])

def upsample(scene, params, opt, integrator):
    use_majorant_supergrid = params.get('use_majorant_supergrid', False)
    for key, _ in params['variables'].items():
        if type(opt[key]) == mi.TensorXf:
            old_res = opt[key].shape
            if len(old_res) in (3,4):
                new_res = (*[2*x for x in old_res[:-1]], old_res[-1])
                opt[key] = dr.upsample(opt[key], shape=new_res)
            else:
                raise ValueError(f"Upsampling expects a 3 or 4D tensor. Got {len(old_res)}.")
            if '.sigma_t.' in key and use_majorant_supergrid:
                adjust_majorant_res_factor(scene, new_res)
        else:
            raise TypeError(f"Upsampling is only supported for mi.TensorXf, got type {type(opt[key])}.")

    integrator.reset()
