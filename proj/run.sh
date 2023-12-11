#!/bin/bash

# python cs285/scripts/run_sweep.py --params_file sweeps/cartpole_linear.json --num_sweeps 2
python cs285/scripts/run_sweep.py --params_file sweeps/cartpole_adaptive_checkpoint.json --num_sweeps 1
python cs285/scripts/run_sweep.py --params_file sweeps/cartpole_sine.json --num_sweeps 2

# python cs285/scripts/run_sweep.py --params_file sweeps/mc_linear.json --num_sweeps 2
# python cs285/scripts/run_sweep.py --params_file sweeps/mc_adaptive.json --num_sweeps 2
# python cs285/scripts/run_sweep.py --params_file sweeps/mc_sine.json --num_sweeps 2

