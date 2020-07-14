# Stock Return Model Comparisons

This project uses GBM, a modified GBM with an Ornstein-Uhlenbeck term, the quantum harmonic oscillator,  builds on results presented by Ahn, Choi and colleagues in “Modeling stock return distribution using the quantum harmonic oscillator”. This project was prepared for Northeastern University's MATH 5131 in the Fall of 218.

## Install

A conda environment for this project can be built using:

`$ conda env create -f environment.yml`

The resulting environment can then be activated: 

`$ conda activate stock-return-model-comparison`

In order to use the notebooks, register the kernel:

`$ python -m ipykernel install --user --name=lume-model-server-demo`

The following command will open the notebook containing the particle swarm optimization code:

`$ jupyter notebook PSO.ipynb`