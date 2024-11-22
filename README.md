# R-FAC
## Introduction
* This is the code repository for the paper "R-FAC: Resilient Value Function Factorization for Multi-Robot Efficient Search with Individual Failure Probabilities" that was submitted to IEEE Trans on Robotics 2023. 
* This is a joint research program among A*STAR, MIT and NUS. (© 2022 A\*STAR. All rights reserved.)


https://github.com/user-attachments/assets/8aba8959-9a8f-4c8a-9e1c-0366565bdfa2


## Dependencies
* Python 3.8
* NetworkX 2.8.4
* Numpy
* Matplotlib
* PyTorch (GPU Acceleration is recommended)
* tqdm

## Description
* We integrate our simulation toolbox into a Python package called `V2DN`. 
* `Train.py` realizes the multiagent reinforcement learning algorithms for given environment and MRS.
* `gym_multi_target.py` constructs training environments with multiple targets. Predefined topological graphs include OFFICE and MUSEUM.
* `Agent.py` defines the structure and actions of individual robots.
* `main.py` claims legal arguments of command line input and the procedure of our algorithm.
* `benchmark` folder contains the state of art algorithm for comparison.

## Usage
If you want to train a MRS of 4 robots in the MUSEUM environment with lr=1e-3 for 500,000, episodes  the command line would be like:
```
python main.py --map_name MUSEUM --robot_num 4 --train_episodes 500000 --lr 1e-3
```
For more operations, please refer to `parse_args()` in `main.py`.
