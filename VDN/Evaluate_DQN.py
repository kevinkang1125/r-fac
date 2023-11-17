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
import copy
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Environment')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Quality_Diversity')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/RL_POMDP')
import rl_utils as rl_utils
from gym_pqh import gym_pqh
from Target import TargetModel
import multi_robot_utils_off_policy as multi_robot_utils_off_policy

class Qnet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.gru = torch.nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, action_mask):
        lengths = torch.tensor([len(seq) for seq in x], dtype=torch.long)
        x = pad_sequence(x, batch_first=True)
        mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.cuda())
        x, _ = self.gru(x.cuda())
        x = x.cuda()
        x = x * mask.unsqueeze(2).float()
        # print("x before lengths - 1:", x)
        x = x[torch.arange(x.size(0)), lengths - 1]
        # print("x after gru:", x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print("after_full_connect:",logits)
        # print("action_mask:",action_mask)
        q_value = x.masked_fill(action_mask, float('-inf'))
        # print("after_action_mask:",logits_masked)
        # return F.softmax(logits_masked, dim=1)
        return q_value

class DQN:
    def __init__(self, obs_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.device = device
        self.action_dim = action_dim
        self.q_net = Qnet(obs_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(obs_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update # target net update frequency
        self.count = 0

    def take_action(self, obs, action_num):
        if np.random.random() < self.epsilon:
            action = np.random.randint(action_num)
        else:
            obs = [obs]
            action_mask = self.create_action_mask(self.action_dim, action_num)
            action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
            action = self.q_net(obs, action_mask).argmax().item()
        return action

    def update(self, transition_dict):
        observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in transition_dict['observations']]
        next_observations = [torch.tensor(next_obs, dtype=torch.float).to(self.device) for next_obs in
                             transition_dict['next_observations']]

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards_part2 = torch.tensor(transition_dict['rewards_part2'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        valid_actions_list = transition_dict['action_num']
        action_masks = self.create_action_masks(self.action_dim, valid_actions_list, self.device)
        next_valid_actions_list = transition_dict['action_num']
        next_action_masks = self.create_action_masks(self.action_dim, next_valid_actions_list, self.device)

        q_values = self.q_net(observations, action_masks).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_observations, next_action_masks).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        #q_joint = q_targets+q_values
        #weights = q_values/q_joint
        #print(q_targets,q_values,q_targets.shape,q_joint*10,weights)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

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

if __name__ == "__main__":
    lr = 5e-2
    epsilon = 0.2
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
    #algo = "V2DN"
    #algo = "VDN"
    
    test_mode = "DUR"#"PRE"#"PRE"
    env_name = "OFFICE"
    horizon = 70 if env_name =="MUSEUM" else 60
    test_steps = 140 if env_name =="MUSEUM" else 120
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
    
    for i in range(robot_num):
        path = "./Benchmark_models/DQN/{}_DQN_R{}_R{}.pth".format(env_name,robot_num,i)
        agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
        agent.q_net = copy.deepcopy(torch.load(path).cuda())
        agents.append(agent)

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
    #print(capture_list)

