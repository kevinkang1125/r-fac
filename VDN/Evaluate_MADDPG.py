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
class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
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
        logits = self.fc2(x)
        # print("after_full_connect:",logits)
        # print("action_mask:",action_mask)
        logits_masked = logits.masked_fill(action_mask, float('-inf'))
        # print("after_action_mask:",logits_masked)
        return F.softmax(logits_masked, dim=1)

class CEPG:
    ''' Vanilla Policy Gradient (REINFORCE) algorithm '''
    def __init__(self, obs_dim, hidden_dim, action_dim, actor_lr, gamma, device, beta):
        self.beta = beta
        self.device = device
        self.action_dim = action_dim
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, obs, action_num, epsilon):

        obs = [obs]
        action_mask = self.create_action_mask(self.action_dim,action_num)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        # print("obs:",obs)
        probs = self.actor(obs, action_mask)
        # print("probs", probs)
        # print("Max value in tensor: {:.2f}".format(max_probs.item()))
        action_dist = torch.distributions.Categorical(probs)
        # print("action_dist:", action_dist)
        action = action_dist.sample()
        # print("action:", action)
        # print("action.item:",action.item())
        return action.item()

    def compute_returns(self, gamma, rewards, rewards_part2, dones):
        n = len(rewards)
        returns = torch.zeros(n, 1)
        next_return = 0
        for t in reversed(range(n)):
            next_return = rewards[t] + gamma * next_return * (1 - dones[t])
            returns[t] = next_return + rewards_part2[t]
        return returns

    def get_probs(self, transition_dict):
        observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in transition_dict['observations']]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        valid_actions_list = transition_dict['action_num']
        action_masks = self.create_action_masks(self.action_dim, valid_actions_list, self.device)
        with torch.no_grad():
            probs = self.actor(observations, action_masks).gather(1, actions)
        return probs

    def update(self, transition_dict):
        observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in transition_dict['observations']]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards_part2 = torch.tensor(transition_dict['rewards_part2'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        valid_actions_list = transition_dict['action_num']
        returns = self.compute_returns(self.gamma, rewards.cpu(), rewards_part2.cpu(), dones.cpu()).to(self.device)
        cross_probs = transition_dict['cross_probs']
        # print('Returns:', returns)
        # action_masks = self.create_action_masks(action_dim, valid_actions_list, self.device)
        # action_masks = [torch.tensor(action_mask, dtype=torch.bool).to(self.device) for action_mask in action_masks]
        action_masks = self.create_action_masks(self.action_dim, valid_actions_list, self.device)
        # print("rewards:",rewards.view(-1))
        # print("rewards_part2:",rewards_part2.view(-1))
        # print("dones:", dones)
        # log_probs = torch.log(self.actor(observations, action_masks).gather(1, actions))
        min_log_prob = -2.303
        max_log_prob = -0.105
        log_probs = torch.clamp(torch.log(self.actor(observations, action_masks).gather(1, actions)), min_log_prob,
                                max_log_prob)
        actor_loss = -torch.mean(log_probs * returns)
        cross_loss = torch.mean(log_probs * cross_probs)
        loss = actor_loss + self.beta * cross_loss
        # print("calculate:", log_probs * returns)
        # print("actor_loss:", actor_loss)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

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

    def clamp(self, x, min_val, max_val):
        return max(min(x, max_val), min_val)

if __name__ == "__main__":
    lr = 5e-2
    epsilon = 0.1
    num_episodes = 80
    target_update = 2
    iter = 100
    
    rho = 0.9
    rho_list = [2,5]
    beta = 0.5
    epoch = 10

    hidden_dim = 256
    gamma = 0.95
    gamma_2 = 0.99
    #device = torch.device("cuda")
    device = torch.device("cuda")
    algo = "V2DN"
    #algo = "VDN"
    
    test_mode = "PRE"#"PRE""DUR"
    env_name = "MUSEUM"
    test_steps = 140 if env_name =="MUSEUM" else 120
    horizon = 70 if env_name =="MUSEUM" else 60
    mode_name = "random"
    robot_num = 3
    target_model = TargetModel("MUSEUM_Random")
    env = gym_pqh(env_name, mode_name, robot_num, target_model)
    torch.manual_seed(0)
    state_dim = env.position_embed
    action_dim = env.action_space
    agents = []
   
    for i in range(robot_num):
        path = "./Benchmark_models/MADDPG/{}_MADDPG_R{}_R{}.pth".format(env_name,robot_num,i)
        agent = CEPG(state_dim, hidden_dim, action_dim, lr, gamma, device, beta)
        agent.actor = copy.deepcopy(torch.load(path))
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
                        action = agent.take_action(obs, action_num,epsilon)
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
                        action = agent.take_action(obs, action_num,epsilon)
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

