import subprocess
import os
from pathlib import Path

subprocess.call(["python", os.path.join(Path(__file__).parents[2], "run_experiment.py"), "ajar", "--output", "gradients", "--n_steps", "21", "--method", "cv_ps"])
