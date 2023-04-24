import subprocess
import os
from pathlib import Path

subprocess.call(["python", os.path.join(Path(__file__).parents[2], "run_experiment.py"), "bunnies", "--method", "cv_ps", "--n_steps", "1000", "--output", "stream"])
