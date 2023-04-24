import os
import multiprocessing as mp
import argparse
import time

from experiments import *

AVAILABLE_METHODS = ["baseline", "cv_ps", "cv_pss"]
DISPATCH_METHODS = ["baseline", "cv_ps"]

def make_video(dir, prefix="img"):
    import subprocess
    dir, method = os.path.split(dir)
    subprocess.run(["ffmpeg", "-i", f"{dir}/{method}/{prefix}/%04d.png", "-y", "-loglevel",  "error", "-c:v:", "libx264", "-movflags", "+faststart",  "-crf", "15", f"{dir}/{method}_{prefix}.mp4"])

def run_experiment(config):
    base_dir = os.path.join(os.path.dirname(__file__), "output")
    if 'output' in config:
        base_dir = os.path.join(base_dir, config['output'])
    else:
        base_dir = os.path.join(base_dir, config['experiment'])

    output_dir = os.path.join(base_dir, config['method'])

    params = globals()[f"scene_{config['experiment']}"](config['method'], output_dir)

    # Override the parameters if necessary
    for key, value in config.items():
        if key not in ['device', 'experiment', 'method']:
            params[key] = value

    result_dict = run_opt(params)
    np.savez(os.path.join(output_dir, "result.npz"), **result_dict)

if __name__ == "__main__":
    # Change the default start method to 'spawn' to avoid CUDA errors
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Run a Mitsuba experiment')
    parser.add_argument('experiment', type=str, nargs='*', choices=AVAILABLE_SCENES, help='Name(s) of the experiment to run')
    parser.add_argument('--method', type=str, choices=AVAILABLE_METHODS, help="Optimization method to use")
    parser.add_argument('--all', action='store_true', help="use all the methods")
    parser.add_argument('--video', action='store_true', help="Generate videos")
    parser.add_argument('--output', type=str, help='Output directory name')

    # Optional overrides of scene parameters
    parser.add_argument('--n_steps', type=int, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--spp', type=int, help='Samples per pixel for the primal rendering')
    parser.add_argument('--spp_grad', type=int, help='Samples per pixel for the adjoint rendering')
    parser.add_argument('--beta1', type=float, help='β₁ parameter for statistics')
    parser.add_argument('--beta2', type=float, help='β₂ parameter for statistics')
    parser.add_argument('--force_baseline', action='store_true', help='Force the use of the baseline primal rendering for gradient computation')
    parser.add_argument('--denoise', action='store_true', help='Denoise the primal rendering')
    parser.add_argument('--loss', type=str, choices=['L1', 'L2', 'VGG'], help='Loss function to use')
    parser.add_argument('--pre_update', action='store_true', help='Update the statistics before computing the control weight')

    args = parser.parse_args()

    print(f"{'Experiment':^20} | {'Method':^20}")
    print(f"{'-'*20:<20} | {'-'*20:<20}")

    # By default, cv_pss is not run, this allows to run it as well
    if args.all:
        methods = AVAILABLE_METHODS
    else:
        methods = DISPATCH_METHODS

    job_queue = []
    for experiment in args.experiment:
        for method in methods if args.method is None else [args.method]:
            job_queue.append({
                'experiment': experiment,
                'method': method,
                **{k: v for k, v in vars(args).items() if v is not None and k not in ['experiment', 'method', 'video', 'all']}
            })

    for config in job_queue:
        process = mp.Process(target=run_experiment, args=(config,))
        process.start()
        process.join()

    # Generate the videos
    if args.video:
        print("Generating videos...")
        for experiment in args.experiment:
            for i, method in enumerate(methods):
                if args.method is not None and args.method != method:
                    continue

                output_dir = os.path.join(os.path.dirname(__file__), 'output', experiment, method)

                make_video(output_dir, 'img') # Main video
                make_video(output_dir, 'img_inf') # Converged video
