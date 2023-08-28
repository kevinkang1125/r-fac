import sys

import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Environment')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Quality_Diversity')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/RL_POMDP')
import benchmark.rl_utils as rl_utils
from gym_pqh_multi_target import gym_pqh
from Target import TargetModel
import benchmark.multi_robot_utils_off_policy as multi_robot_utils_off_policy
from benchmark.ReplayBuffer import ReplayBuffer

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
        mask = mask.to(x.device)
        x, _ = self.gru(x)
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

class V2DN_agent:
    def __init__(self,obs_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
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

    def update(self, transition_dict,td_errors):
        observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in transition_dict['observations']]
        next_observations = [torch.tensor(next_obs, dtype=torch.float).to(self.device) for next_obs in
                             transition_dict['next_observations']]

        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards_part2 = torch.tensor(transition_dict['rewards_part2'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        valid_actions_list = transition_dict['action_num']
        action_masks = self.create_action_masks(self.action_dim, valid_actions_list, self.device)
        # next_valid_actions_list = transition_dict['action_num']
        # next_action_masks = self.create_action_masks(self.action_dim, next_valid_actions_list, self.device)

        q_values = self.q_net(observations, action_masks).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        # max_next_q_values = self.target_q_net(next_observations, next_action_masks).max(1)[0].view(
        #     -1, 1)
        q_targets = q_values.clone().detach() + td_errors  # TD误差目标
        ##use TD error 
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
    
class V2DN:
    def __init__(self,n_agents, input_dim, n_actions):
        self.agents = [V2DN_agent(input_dim, n_actions) for _ in range(n_agents)]
    def learn(self, states, actions, team_reward, next_states,alive_index):
        # Get Q-values for the current state and action for all agents
        current_q_values = [agent.model(torch.FloatTensor(state))[action].item() for agent, state, action in zip(self.agents, states, actions)]
        
        # For the next state, get max Q-values of next states from all agents
        next_max_q_values = [torch.max(agent.model(torch.FloatTensor(next_state))).item() for agent, next_state in zip(self.agents, next_states)]
        
        # Compute the joint target Q-value using the team reward
        joint_target_q_value = team_reward + self.agents[0].gamma * sum(next_max_q_values)
        
        # Calculate TD errors for individual agents
        td_errors = [joint_target_q_value - q for q in current_q_values]
        
        # Update each agent's Q-network based on its TD error
        for i, (agent, state, action, next_state) in enumerate(zip(self.agents, states, actions, next_states)):
            agent.learn(state, action, td_errors[i])
            

if __name__ == "__main__":
    lr = 1e-4
    epsilon = 0.01
    num_episodes = 20000
    target_update = 10
    buffer_size = 1000
    minimal_size = 500
    batch_size = 128

    hidden_dim = 128
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
    replay_buffers = []

    for i in range(robot_num):
        agent = V2DN_agent(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
        replay_buffer = ReplayBuffer(buffer_size)
        agents.append(agent)
        replay_buffers.append(replay_buffer)

    return_list = multi_robot_utils_off_policy.train_off_policy_multi_agent(env, agents, replay_buffers, num_episodes,
                                                                            minimal_size, batch_size)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('VPG on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 101)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('VPG on {}'.format(env_name))
    plt.show()