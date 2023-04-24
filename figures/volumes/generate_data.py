import subprocess
import os
from pathlib import Path
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

scenes = ["dust_devil", "janga", "rover"]
spp_high = [128, 32, 32]

for scene, spp in zip(scenes, spp_high):
    # Equal budget
    subprocess.call(["python", os.path.join(Path(__file__).parents[2], "run_experiment.py"), scene])
    # Run baseline at higher spp (equal quality)
    subprocess.call(["python", os.path.join(Path(__file__).parents[2], "run_experiment.py"), scene, "--spp", f"{spp}", "--method", "baseline", "--output", f"{scene}_high_spp"])

subprocess.call(["python", os.path.join(os.path.dirname(__file__), "render_dust.py")])