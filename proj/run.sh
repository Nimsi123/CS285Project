#!/bin/bash

python cs285/scripts/run_sweep.py --name two --params_file sweeps/cartpole_linear.json --num_sweeps 1
python cs285/scripts/run_sweep.py --name one --params_file sweeps/cartpole_adaptive.json --num_sweeps 1
python cs285/scripts/run_sweep.py --name two --params_file sweeps/mc_linear.json --num_sweeps 1
python cs285/scripts/run_sweep.py --name one --params_file sweeps/mc_adaptive.json --num_sweeps 1