import subprocess
import os
from pathlib import Path

subprocess.call(["python", os.path.join(Path(__file__).parents[2], "run_experiment.py"), "bunnies", "--all", "--force_baseline", "--output", "weights", "--n_steps", "50"])
