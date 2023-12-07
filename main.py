import sys

import random
import gym
import numpy as np
import collections
from tqdm import tqdm
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


if __name__ == "__main__":
    lr = 2e-5
    epsilon = 0.12
    num_episodes = 30000
    target_update = 5
    iter = 10
     
    rho = 0.08
    hidden_dim = 256
    gamma = 0.95
    gamma_2 = 0.9
    device = torch.device("cuda")
    algo = "V2DN"
    

    #env_name = "MUSEUM"
    env_name = "OFFICE"
    horizon = 70 if env_name =="MUSEUM" else 60
    mode_name = "random"
    robot_num = 5
    target_model = TargetModel("OFFICE_Random")
    env = gym_search(env_name, mode_name, robot_num, target_model)
    torch.manual_seed(0)
    state_dim = env.position_embed
    action_dim = env.action_space
    agents = []
    replay_buffers = []
    if robot_num == 3:
        if rho == 0.04:
            rho_list = [8,21]
        elif rho == 0.06:
            rho_list = [6,14]
        else:
            rho_list = [4,10]
    elif robot_num == 4:
        if rho == 0.04:
            rho_list = [6,15,28]
        elif rho == 0.06:
            rho_list = [4,10,18]
        else:
            rho_list = [3,7,13]
    elif robot_num == 5: 
        if rho == 0.04:
            rho_list = [5,11,20,33]
        elif rho == 0.06:
            rho_list = [3,7,13,21]
        else:
            rho_list = [2,6,9,16]

    for i in range(robot_num):
        agent = Robot(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
        agents.append(agent)
    

    #mixer = V2DN_pre(gamma_2,agent_num=robot_num, horizon= horizon)
    mixer = V2DN_dur(gamma_2,robot_num, horizon= horizon)


    #return_list, td_list = rf.train_resilient_on_policy_multi_agent(env, mixer, agents, num_episodes, rho, iter)
    return_list, td_list = rf.train_resilient_on_policy_multi_agent_dur(env, mixer, agents, num_episodes, rho_list, iter)
    # for i in range(robot_num):
    #     agents[i].save('./on policy robot{} in teamsize{} with rho{} in {}.pth'.format(i,robot_num,rho,env_name))
    # episodes_list = list(range(len(return_list)))
    
    for h in range(len(agents)):
        net_name = "./Benchmark_models/V2DN/Dur/0.006/" + env_name + "_V2DN_R" + str(len(agents)) + "_R" + str(h)
        torch.save(agents[h].q_net, net_name + '.pth')
    # for h in range(len(agents)):
    #     net_name = "./Benchmark_models/V2DN/" + env_name + "_V2DN_dur_R" + str(len(agents)) + "_R" + str(h)
    #     torch.save(agents[h].q_net, net_name + '.pth')
    #plt.plot(episodes_list, return_list)
    
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