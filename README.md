# Framework for Model (order) reduction on manifolds
A framework for MOR on manifolds to reproduce the online phase of the experiments in

- Buchfink, Patrick and Glas, Silke and Haasdonk, Bernard 
- Symplectic Model Reduction of Hamiltonian Systems on Nonlinear Manifolds and Approximation with Weakly Symplectic Autoencoder 
- 2023 
- SIAM Journal on Scientific Computing, Vol. 45, No. 2, p. A289-A311

## Installation (Linux)
To guarantee exact reproduction of the code (a) install Python3.7 and (b) install the package (and dependencies) via pip.

### Installation of python3.7 and virtual environment
Install python3.7, e.g. in Ubuntu

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7 -y
sudo apt install python3.7-venv -y
sudo apt-get install python3.7-tk -y
```

Generate a virtual environment in python3.7

```
python3.7 -m venv ~/.venvs/experiments-manifold-mor
source ~/.venvs/experiments-manifold-mor/bin/activate
```

### Installation of package via pip
Make sure that the virtual environment is active. Then change to the folder `manifold-mor-wave` and run

```
pip install pip==23.3.1
pip install -e .
pip install torch==1.7.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

### Install package for Hamiltonian models
Install the package containing the Hamiltonian models from the URL `https://github.com/pbuchfink/hamiltonian-models`

### Alternative installation
Alternatively, you may try to install newer versions of python and dependencies. You have to modify (a) the required python version and (b) the versions of dependencies in `pyproject.toml` before installing the package with pip.

## Running online phase of wave experiment
In the demos folder, two scripts are provided, are used to reproduce the online phase of the results.

In the main folder run

```
python src/manifold_mor_demos/hamiltonian_wave/main_online_phase_paper.py
```

or for parallel execution with NUMBER_OF_CPUS cpus

```
python src/manifold_mor_demos/hamiltonian_wave/main_online_phase_paper.py --ncpu NUMBER_OF_CPUS
```

As soon as the script is finished, the plots can be generated with

```
python src/manifold_mor_demos/hamiltonian_wave/generate_plots_for_paper_from_bump_online.py
```