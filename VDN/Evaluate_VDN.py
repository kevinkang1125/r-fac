import sys

import random
import gym
import numpy as np
import copy
import collections
from tqdm import tqdm
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Environment')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Quality_Diversity')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/RL_POMDP')
import rl_utils as rl_utils
from gym_pqh import gym_pqh
from Target import TargetModel
import multi_robot_utils_off_policy as multi_robot_utils_off_policy
#import V2DN_off_policy
#from V2DN_off_policy import Agent as V2DN_Agent
class Qnet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim,device):
        super(Qnet, self).__init__()
        self.device = device
        self.gru = torch.nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, action_mask):
        lengths = torch.tensor([len(seq) for seq in x], dtype=torch.long)
        x = pad_sequence(x, batch_first=True)
        mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.cuda())
        x = x.cuda()
        #mask = mask.to(x)
        #x = x.cuda()
        x, _ = self.gru(x)
        x = x * mask.unsqueeze(2).float()
        x = x[torch.arange(x.size(0)), lengths - 1]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        q_value = x.masked_fill(action_mask, float('-inf'))
        return q_value

class Agent:
    def __init__(self,obs_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, device):
        self.device = device
        self.action_dim = action_dim
        self.q_net = Qnet(obs_dim, hidden_dim, action_dim,device).to(device)
        self.target_q_net = Qnet(obs_dim, hidden_dim, action_dim,device).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update # target net update frequency
        #self.loss_fn = F.mse_loss(reduction='sum')
        self.count = 0
        self.count_list = []

    def take_action(self, obs, action_num):
        if np.random.random() < self.epsilon:
            action = np.random.randint(action_num)
        else:
            obs = [obs]
            action_mask = self.create_action_mask(self.action_dim, action_num)
            action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
            action = self.q_net(obs, action_mask).argmax().item()
        return action

    def create_action_mask(self, total_actions, valid_actions):
        action_mask = [False] * valid_actions + [True] * (total_actions - valid_actions)
        return action_mask

    def create_action_masks(self, total_actions, valid_actions_list, device):
        action_masks = []
        for valid_actions in valid_actions_list:
            action_mask = [False] * valid_actions + [True] * (total_actions - valid_actions)
            action_masks.append(action_mask)
        action_masks_tensor = torch.tensor(action_masks, dtype=torch.bool).to(device)
        return action_masks_tensor
    
    def save(self,path):
        torch.save(self.q_net.state_dict(),path)
    def load(self,path):
        self.q_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
 


if __name__ == "__main__":
    lr = 5e-2
    epsilon = 0.1
    num_episodes = 80
    target_update = 2
    iter = 100
    
    rho = 0.04
    rho_list = [2,5]
    beta = 0.5
    epoch = 10

    hidden_dim = 128
    gamma = 0.95
    gamma_2 = 0.99
    #device = torch.device("cuda")
    device = torch.device("cuda")
    algo = "V2DN"
    #algo = "VDN"
    # 4 museum r10 t3 *1.
    test_mode = "DUR"#"PRE""DUR"
    env_name = "OFFICE"
    test_steps = 140 if env_name =="MUSEUM" else 120
    horizon = 70 if env_name =="MUSEUM" else 60
    mode_name = "random"
    robot_num = 3
    target_model = TargetModel("OFFICE_Random")
    env = gym_pqh(env_name, mode_name, robot_num, target_model)
    torch.manual_seed(0)
    state_dim = env.position_embed
    action_dim = env.action_space
    agents = []
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

    # for i in range(robot_num):
    #     agent = Agent(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    #     agents.append(agent)
    
    for i in range(robot_num):
        path = "./Benchmark_models/VDN/OFFICE_VDN_R{}_R{}.pth".format(robot_num,i)
        agent = Agent(state_dim, hidden_dim, action_dim, lr, gamma,epsilon, device)
        agent.q_net = copy.deepcopy(torch.load(path).cuda())
        agents.append(agent)

    # for i in range(robot_num):
    #     agents[i].load('./off policy robot{} in teamsize{} with rho{} in {}.pth'.format(i,robot_num,rho,env_name))
        #agents[i].load('./off policy robot{} in teamsize{} with rho{} in {}.pth'.format(i,robot_num,rho,env_name))
    
    epoch_list = []
    if test_mode == "PRE":
        
        for num in range(epoch):
            record_list = []
            for iteration in tqdm(range(iter),desc="progress"):
                team_done = False
                episode_return = 0.0
                agent_num = robot_num
                transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                                    'dones': [], 'action_num': []} for _ in range(agent_num)]
                alive_list = []
                ##faulty sample
                for i in range(agent_num):
                    if np.random.random() < rho:
                        alive_list.append(i)
                if len(alive_list)== 0:
                    team_done = True
                    counter = 140 if env_name =="MUSEUM" else 120
                else:
                    observations, states, action_nums = env.reset()
                    counter = 0
                while not team_done:
                    env._target_move()
                    agent_done_list = []
                    for m in (alive_list):
                        transition_dict = transition_dicts[i]
                        if counter == 0:
                            obs = observations[i]
                        else:
                            obs = transition_dict['next_observations'][-1]
                        agent = agents[i]
                        action_num = action_nums[i]
                        action = agent.take_action(obs, action_num)
                        transition_dict['action_num'].append(action_num)
                        next_obs, next_state, reward, done, action_num = env.step(action, i)
                        action_nums[i] = action_num
                        transition_dict['observations'].append(obs)
                        transition_dict['actions'].append(action)
                        transition_dict['next_observations'].append(next_obs)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        #transition_dict['rewards_part2'].append(reward_part2)
                        transition_dict['dones'].append(done)
                        agent_done_list.append(done)
                    counter += 1
                    team_done = any(agent_done_list)
                    if counter == test_steps:
                        team_done = True
                record_list.append(counter)
            # capture_list = []
            # for m in range(len(record_list)):
            #     if record_list[m] < test_steps:
            #         capture_list.append(record_list[m])

            # print(record_list,capture_list,sum(capture_list)/len(capture_list))
            # epoch_list.append(sum(capture_list)/len(capture_list))
            print(record_list,sum(record_list)/len(record_list))
            epoch_list.append(sum(record_list)/len(record_list))
        print(epoch_list,sum(epoch_list)/len(epoch_list))
    elif test_mode == "DUR":
        for num in range(epoch):
            record_list = []
            for iteration in tqdm(range(iter),desc="progress"):
                team_done = False
                agent_num = robot_num
                transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                                    'dones': [], 'action_num': []} for _ in range(agent_num)]
                ##faulty sample
                alive_index = []
                for m in range(agent_num):
                    alive_index.append(m)
                observations, states, action_nums = env.reset()
                counter = 0
                while not team_done:
                    agent_done_list = []
                    env._target_move()
                    if counter in rho_list:
                        robot = random.choice(alive_index)
                        alive_index.remove(robot)

                    # obs_list = env.observation_list
                    for i in (alive_index):
                        transition_dict = transition_dicts[i]
                        if counter == 0:
                            obs = observations[i]
                        else:
                            obs = transition_dict['next_observations'][-1]
                        agent = agents[i]
                        action_num = action_nums[i]
                        action = agent.take_action(obs, action_num)
                        transition_dict['action_num'].append(action_num)
                        next_obs, next_state, reward, done, action_num = env.step(action, i)
                        action_nums[i] = action_num
                        transition_dict['observations'].append(obs)
                        transition_dict['actions'].append(action)
                        transition_dict['next_observations'].append(next_obs)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        #transition_dict['rewards_part2'].append(reward_part2)
                        transition_dict['dones'].append(done)
                        agent_done_list.append(done) 
                    counter += 1
                    team_done = any(agent_done_list)
                    if counter == test_steps:
                        team_done = True
                record_list.append(counter)
            print(record_list,sum(record_list)/len(record_list))
            epoch_list.append(sum(record_list)/len(record_list))
        print(epoch_list,sum(epoch_list)/len(epoch_list))
                    #alive_lists.append(alive_index.copy())
    #print(capture_list)






       



    # plt.plot(td_list) 
    # plt.xlabel('Episodes')
    # plt.ylabel('TD_error')
    # plt.title('Performance on {} with rho={} with {}'.format(env_name,rho,algo))
    # plt.show()
    # np.savetxt("td_list.txt",td_list)
    # mv_return = rl_utils.moving_average(td_list, 101)
    # plt.plot(mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('average_td')
    # plt.title('VPG on {}'.format(env_name))
    # plt.show()

    # plt.plot(td_list) 
    # plt.xlabel('Episodes')
    # plt.ylabel('TD_error')
    # plt.title('Performance on {} with rho={} with {}'.format(env_name,rho,algo))
    # plt.show()
    # np.savetxt("td_list.txt",td_list)