import subprocess
import os
from pathlib import Path

scene_name = "ajar"
# L1
subprocess.call(["python", os.path.join(Path(__file__).parents[2], "run_experiment.py"), scene_name, "--lr", "0.02", "--output", f"{scene_name}_l1"])
subprocess.call(["python", os.path.join(Path(__file__).parents[2], "run_experiment.py"), scene_name, "--lr", "0.02", "--denoise", "--output", f"{scene_name}_l1_denoised"])

# L2
subprocess.call(["python", os.path.join(Path(__file__).parents[2], "run_experiment.py"), scene_name, "--loss", "L2", "--lr", "0.01", "--output", f"{scene_name}_l2"])
subprocess.call(["python", os.path.join(Path(__file__).parents[2], "run_experiment.py"), scene_name, "--loss", "L2", "--lr", "0.01", "--denoise", "--output", f"{scene_name}_l2_denoised"])
