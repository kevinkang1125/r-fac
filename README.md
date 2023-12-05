# R-FAC
## Introduction
* This is the code repository for the paper "R-FAC: Resilient Value Function Factorization for Multi-Robot Efficient Search with Individual Failure Probabilities" that was submitted to IEEE Trans on Robotics 2023. 
* This is a joint research program among A*STAR, MIT and NUS. (© 2022 A\*STAR. All rights reserved.)

## Dependencies
* Python 3.8
* NetworkX 2.8.4
* Numpy
* Matplotlib
* PyTorch (GPU Acceleration is recommended)
* tqdm

## Description
* We integrate our simulation toolbox into a Python package called `V2DN`. 
* `opt.py` realizes the centralized optimizer for given environment and MRS.
* `env.py` constructs environments. Predefined topological graphs include OFFICE and MUSEUM.
* `agent.py` defines the structure and actions of individual robots.
* `main.py` claims legal arguments of command line input and the procedure of our algorithm.

## Usage
If you want to optimize a MRS of 4 robots in the MUSEUM environment with lr=1e-3 for 5,000 epochs and test it for 1,000,000 steps and require real-time feedbacks and final outputs, the command line would be like:
```
python main.py --env MUSEUM --size_MRS 4 --epoch 5000 --step 1000000 --lr 1e-3 --verbose --output
```
For more operations, please refer to `parse_opt()` in `main.py`.
