
<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center"><a href="http://rgl.epfl.ch/publications/Nicolet2023Recursive">Recursive Control Variates for Inverse Rendering</a></h1>

  <a href="http://rgl.epfl.ch/publications/Nicolet2023Recursive">
    <img src="http://rgl.s3.eu-central-1.amazonaws.com/media/images/papers/Nicolet2023Recursive_teaser.jpg" alt="Logo" width="100%">
  </a>

  <p align="center">
    ACM Transactions on Graphics (Proceedings of SIGGRAPH), July 2023.
    <br />
    <a href="https://bnicolet.com/"><strong>Baptiste Nicolet</strong></a>
    ·
    <a href="https://research.nvidia.com/person/fabrice-rousselle"><strong>Fabrice Rousselle</strong></a>
    ·
    <a href="https://jannovak.info/"><strong>Jan Novák</strong></a>
    ·
    <a href="https://research.nvidia.com/person/alex-keller"><strong>Alexander Keller</strong></a>
    ·
    <a href="https://rgl.epfl.ch/people/wjakob"><strong>Wenzel Jakob</strong></a>
    ·
    <a href="https://tom94.net/"><strong>Thomas Müller</strong></a>
  </p>

  <p align="center">
    <a href='http://rgl.s3.eu-central-1.amazonaws.com/media/papers/Nicolet2023Recursive_1.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
    <a href='http://rgl.epfl.ch/publications/Nicolet2023Recursive' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat-square' alt='Project Page'>
    </a>
  </p>
</p>

<br />
<br />

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 15px; border-radius:5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#overview">Overview</a>
    <li><a href="#citation">Citation</a>
    <li><a href="#getting-started">Getting started</a>
    <li><a href="#running-an-optimization">Running an optimization</a>
    <li><a href="#reproducing-figures">Reproducing figures</a>
    <li><a href="#using-control-variates">Using control variates</a>
    <li><a href="#acknowledgements">Acknowledgements</a>
  </ol>
</details>

Overview
--------

This repository contains code examples to reproduce the results from the article:

> Baptiste Nicolet and Fabrice Rousselle and Jan Novák and Alexander Keller and Wenzel Jakob and Thomas Müller, 2023.
> Recursive Control Variates for Inverse Rendering.
> In Transactions on Graphics (Proceedings of SIGGRAPH) 42(4).

It uses the [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) differentiable renderer.


Citation
--------

This code is released under the [BSD 3-Clause License](LICENSE). Additionally, if you are using this code in academic research, please cite our paper using the following BibTeX entry:

```bibtex
@article{Nicolet2023Recursive,
    author = {Baptiste Nicolet and Fabrice Rousselle and Jan Novák and Alexander Keller and Wenzel Jakob and Thomas Müller},
    title = {Recursive Control Variates for Inverse Rendering},
    journal = {Transactions on Graphics (Proceedings of SIGGRAPH)},
    volume = {42},
    number = {4},
    year = {2023},
    month = aug,
    doi = {10.1145/3592139}
}
```


Getting started
---------------

This code was tested on Ubuntu 22.04 with an NVIDIA RTX 4090 GPU.
NVIDIA driver version 525.60.13 was used with CUDA 12.0.

Mitsuba 3 was compiled with Clang++ 11.0.0 and the provided scripts were run with Python 3.9.12.
The `cuda_ad_rgb` Mitsuba variant was selected, although the `llvm_ad_rgb` variant is also compatible in principle.

This implementation relies on modifications to the Mitsuba source code, which are available on the `unbiased-volume-opt` branch of the `mitsuba3` repository.
**Please make sure to checkout the correct branch** as follows.
Note the `--recursive` and `--branch` flags:

```bash
# Cloning Mitsuba 3 and this repository
git clone --recursive https://github.com/mitsuba-renderer/mitsuba3 --branch unbiased-volume-opt
git clone --recursive https://github.com/rgl-epfl/recursive_control_variates

# Building Mitsuba 3, including the project-specific modifications
cd mitsuba3
mkdir build && cd build
cmake -GNinja ..
ninja
```

The `cuda_ad_rgb` and `llvm_ad_rgb` variants should be included by default.
Please see the [Mitsuba 3 documentation](https://mitsuba.readthedocs.io/en/latest/#) for complete instructions on building and using the system.

The scene data must be downloaded and unzipped at the root of the project folder:

```bash
cd recursive_control_variates
wget https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Nicolet2023Recursive.zip
unzip Nicolet2023Recursive.zip
rm Nicolet2023Recursive.zip
ls scenes
# The available scenes should now be listed (one directory per scene)
```

Running an optimization
-----------------------

Navigate to this project's directory and make sure that the Mitsuba 3 libraries built in the previous step are made available in your current session using `setpath.sh`:

```bash
cd recursive_control_variates
source ../mitsuba3/build/setpath.sh
# The following should execute without error and without output
# (use the variant 'llvm_ad_rgb' if your system does not support the CUDA backend):
python3 -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb')"
```

From here, the script `python/run_experiment.py` can be used to run inverse rendering examples using different methods:

```bash
python3 run_experiment.py teapot --method cv_ps
```

The script expects a scene name (`teapot` in the example above). The scene
configurations are defined in `experiments.py`. You can also specify with which method you want to run the optimisation:
- `baseline`: Optimisation with standard differentiable rendering integrators.
- `cv_pss`: Optimisation using our control variates, with the 'primary sample space' implementation (see the paper for an explanation).
- `cv_ps`: Optimisation using our control variates, with the 'path space' implementation.

If no method is specified, both `baseline` and `cv_ps` optimisations will be run.

Reproducing figures
-------------------

We provide scripts to reproduce the figures from the paper. These are located in
the `figures` subfolder. Each figure has its own subfolder, with a
`generate_data.py`, that will run the relevant experiments needed to generate
the figure. Then, one can run the `figure.ipynb` notebook to generate the figure.


Using control variates
---------------------------------------------

In [`tutorial.ipynb`](tutorial.ipynb), we show how our control variates can be easily integrated
into an optimisation. One merely needs to use the `TwoState` adapter BSDF (resp.
medium) for the surface (resp. medium) being optimised, and use the `twostate`
variant of the `prb` (resp. `prbvolpath`) integrator.


Acknowledgements
----------------
This README template was created by [Miguel Crespo](https://github.com/mcrescas/viltrum-mitsuba/blob/457a7ffbbc8b8b5ba9c40d6017b5d08f0f41a886/README.md), and its structure inspired by [Merlin Nimier-David](https://github.com/rgl-epfl/unbiased-inverse-volume-rendering/blob/master/README.md).
Many figure generating utility functions were written by [Delio Vicini](https://dvicini.github.io).

The `volpathsimple` integrator was implemented by [Merlin Nimier-David](https://github.com/rgl-epfl/unbiased-inverse-volume-rendering)

Volumes, environment maps and 3D models were generously provided by [JangaFX](https://jangafx.com/software/embergen/download/free-vdb-animations/), [PolyHaven](https://polyhaven.com/hdris), [Benedikt Bitterli](https://benedikt-bitterli.me/resources/) and [vajrablue](https://blendswap.com/blend/28458).
