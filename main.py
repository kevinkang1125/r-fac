import sys

import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import argparse
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from V2DN.Agent import Robot
from V2DN import rl_utils
from V2DN.gym_multi_target import gym_search
from V2DN.Target import TargetModel
from V2DN.ReplayBuffer import ReplayBuffer
import V2DN.R_Fac_train as rf 
from V2DN.Train import V2DN_dur,V2DN_pre


parser = argparse.ArgumentParser(description='R-FAC_V2DN')
parser.add_argument('--map_name', default='Museum',
                    help='Map_Name')
parser.add_argument('--mode_name', default='random',
                    help='Target moving policy')
parser.add_argument('--target_model', default= TargetModel("MUSEUM_Random"),
                    help='Target Map and Policy')
parser.add_argument('--result_dir', './results',
                    help="Directory Path to store results")
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Enforces no cuda usage (default: %(default)s)')
parser.add_argument('--train', action='store_true', default=True,
                    help='Trains the model')
parser.add_argument('--n_inter', default=10,
                    help='The number of intervals.')
parser.add_argument('--robot_num', default=3,
                    help='The number of agents.')
parser.add_argument('--failure', default='PRE',
                    help='failure type of robots(PRE refer pre-deployment and DUR refer to during execution')
parser.add_argument('--rho', default=0.9,
                    help='failure rate of robot')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='Learning rate')
parser.add_argument('--discount', type=float, default=0.9,
                    help=' Discount rate (or Gamma) for TD error (default: %(default)s)')
parser.add_argument('--train_episodes', type=int, default=10000,
                    help='episodes for training (default: %(default)s)')
parser.add_argument('--batch_size', type=int, default=70,
                    help='Policy horizon')
parser.add_argument('--seed', type=int, default=0,
                    help='seed (default: %(default)s)')
args = parser.parse_args()

if __name__ == "__main__":
    lr = args.lr
    epsilon = 0.12
    num_episodes = args.train_episodes
    target_update = 5
    iter = 10
     
    rho = 0.08
    hidden_dim = 256
    #gamma = 0.95
    gamma_2 = args.discount
    device = torch.device("cuda")
    algo = "V2DN"
    failure_mode = "PRE"

    env_name = args.map_name
    horizon = 70 if env_name =="MUSEUM" else 60
    mode_name = "random"
    robot_num = args.robot_num
    target_model = args.target_model
    env = gym_search(env_name, mode_name, robot_num, target_model)
    torch.manual_seed(0)
    state_dim = env.position_embed
    action_dim = env.action_space
    agents = []
    rho_list = rl_utils.rho_transfer(rho,robot_num)

    for i in range(robot_num):
        agent = Robot(state_dim, hidden_dim, action_dim, lr, gamma_2, epsilon, target_update, device)
        agents.append(agent)
    
    if failure_mode == "PRE":
        mixer = V2DN_pre(gamma_2,agent_num=robot_num, horizon= horizon)
        return_list, td_list = rf.train_resilient_on_policy_multi_agent(env, mixer, agents, num_episodes, rho, iter)
    elif failure_mode == "DUR":
        mixer = V2DN_dur(gamma_2,robot_num, horizon= horizon)
        return_list, td_list = rf.train_resilient_on_policy_multi_agent_dur(env, mixer, agents, num_episodes, rho_list, iter)

    
    for h in range(len(agents)):
        net_name = "./results/" +failure_mode +rho+ env_name + "_V2DN_R" + str(len(agents)) + "_R" + str(h)
        torch.save(agents[h].q_net, net_name + '.pth')
    
    plt.subplot(221)
    plt.plot(return_list) 
    plt.xlabel('Episodes')
    plt.ylabel('team_reward')
    plt.title('On-policy DUR reward on {} with rho={} with {}'.format(env_name,rho,robot_num))
    #plt.show()
    plt.subplot(222)
    plt.plot(td_list) 
    plt.xlabel('Times')
    plt.ylabel('TD_error')
    plt.title('On-policy TD_error on {} with rho={} with {}'.format(env_name,rho,algo))
    #plt.show()
    np.savetxt("td_list.txt",td_list)
    mv_return = rl_utils.moving_average(return_list, 101)
    plt.subplot(223)
    plt.plot(mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('average_reward')
    plt.title('On-policy DUR average reward on {}'.format(env_name))
    #plt.show()
    mv_return = rl_utils.moving_average(td_list, 101)
    plt.subplot(224)
    plt.plot(mv_return)
    plt.xlabel('Times')
    plt.ylabel('average_td')
    plt.title('On-policy average TD-Error on {}'.format(env_name))
    plt.show()