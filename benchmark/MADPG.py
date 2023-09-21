# MADPG efficient search version

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.utils.rnn import pad_sequence
import random
import sys

sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Environment')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Quality_Diversity')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/RL_POMDP')
import rl_utils as rl_utils
from gym_pqh_multi_target import gym_pqh
from Target import TargetModel
import multi_robot_utils_archive_4th

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
        x = x.cuda()
        x, _ = self.gru(x)
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

class MADPG:
    ''' Vanilla Policy Gradient (REINFORCE) algorithm '''
    def __init__(self, obs_dim, hidden_dim, action_dim, actor_lr, gamma, device):
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

    def update(self, transition_dict):
        observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in transition_dict['observations']]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards_part2 = torch.tensor(transition_dict['rewards_part2'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        valid_actions_list = transition_dict['action_num']
        returns = self.compute_returns(self.gamma, rewards.cpu(), rewards_part2.cpu(), dones.cpu()).to(self.device)
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
        # print("log_probs:", log_probs)
        actor_loss = -torch.mean(log_probs * returns)
        # print("calculate:", log_probs * returns)
        # print("actor_loss:", actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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
    actor_lr = 2e-5
    diversity_lr = 2e-5
    num_episodes = 40
    hidden_dim = 256
    gamma = 0.95
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = "MUSEUM"
    mode_name = "random"
    robot_num = 3
    target_model = TargetModel("MUSEUM_Random")
    env = gym_pqh(env_name, mode_name, robot_num, target_model)
    torch.manual_seed(0)
    state_dim = env.position_embed
    action_dim = env.action_space

    agents = []
    for i in range(robot_num):
        agent = MADPG(state_dim, hidden_dim, action_dim, actor_lr, gamma, device)
        agents.append(agent)

    return_list = multi_robot_utils_archive_4th.train_on_policy_multi_agent_MADPG(env, agents, num_episodes)
    #return_list = multi_robot_utils_archive_4th.train_on_policy_multi_agent(env, agents, num_episodes)
    for h in range(len(agents)):
        env_name = env.env_name
        net_name = "benchmark_save_model/" + env_name + "_MADDPG_R" + str(len(agents)) + "_R" + str(h)
        torch.save(agents[h].q_net, net_name + '.pth')

    episodes_list = list(range(len(return_list)))
    np.savetxt("MaDDPG",return_list)
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('MADPG on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 101)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('MADPG on {}'.format(env_name))
    plt.show()
