import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from Efficient_Search.RL_POMDP import rl_utils
from Efficient_Search.Environment.gym_pqh_multi_target import gym_pqh
from Efficient_Search.Environment.Target import TargetModel
from Efficient_Search.Quality_Diversity import multi_robot_utils_off_policy
from ReplayBuffer import ReplayBuffer

class D_Qnet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, num_atoms):
        super(D_Qnet, self).__init__()
        self.gru = torch.nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim * num_atoms)
        self.num_atoms = num_atoms

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
        # print("action_mask:", action_mask)
        q_value_dist = x.view(x.size(0), -1, self.num_atoms)  # reshape to batch x actions x atoms
        q_value_dist = F.softmax(q_value_dist, dim=2)  # apply softmax over atoms dimension
        # print("q_value_soft:",q_value_dist)
        q_value_dist = q_value_dist.masked_fill(action_mask.unsqueeze(-1), 0)  # apply mask over action dimension
        # print("q_value_after:",q_value_dist)
        return q_value_dist

class D_DQN:
    def __init__(self, obs_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                 num_atoms=21, vmin = -2., vmax=8.):
        self.device = device
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = vmin
        self.v_max = vmax
        self.atoms = torch.linspace(vmin, vmax, self.num_atoms).to(device)
        self.delta_z = (vmax - vmin)/(self.num_atoms - 1)

        self.q_net = D_Qnet(obs_dim, hidden_dim, action_dim, num_atoms).to(device)
        self.target_q_net = D_Qnet(obs_dim, hidden_dim, action_dim, num_atoms).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update # target net update frequency
        self.count = 0

    def take_max_action(self, obs, action_num):
        obs = [obs]
        action_mask = self.create_action_mask(self.action_dim, action_num)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        action_dist = self.q_net(obs, action_mask).view(-1, self.action_dim, self.num_atoms)
        q_expectations = torch.sum(action_dist * self.atoms, dim=2)
        q_expectations = q_expectations[:, :action_num]
        action = q_expectations.argmax().item()
        # if action >= action_num:
        #     print("mistake1")
        return action

    # def take_max_action_list(self, next_obs, next_action_masks):


    def take_action(self, obs, action_num):
        if np.random.random() < self.epsilon:
            action = np.random.randint(action_num)
        else:
            action = self.take_max_action(obs, action_num)
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
        next_valid_actions_list = transition_dict['next_action_num']
        next_action_masks = self.create_action_masks(self.action_dim, next_valid_actions_list, self.device)

        q_dist = self.q_net(observations, action_masks)
        next_q_dist = self.q_net(next_observations, next_action_masks)
        next_q_dist_target = self.target_q_net(next_observations, next_action_masks)
        # print("q_dist.size", q_dist.size())
        # print("q_dist.size", q_dist.sum(-1).sum())
        q_expectations = torch.sum(next_q_dist * self.atoms, dim=2)
        masked_q_expectations = q_expectations.masked_fill(next_action_masks, float('-inf'))
        next_actions = masked_q_expectations.argmax(1).unsqueeze(1)
        # next_actions = self.take_max_action(next_observations, next_valid_actions_list)
        # print("actions.size()",actions.size())
        # print("next_actions.size()",next_actions.size())
        # print("next_actions:",next_actions)
        # next_actions = next_q_dist.sum(2).argmax(1)  # select actions based on main network
        # next_q_dist_sum = next_q_dist.sum(2)
        # masked_next_q_dist = next_q_dist_sum.masked_fill(~next_action_masks, float('-inf'))
        # next_actions = masked_next_q_dist.argmax(1)
        # print("next_actions:", next_actions)
        next_q_dist = next_q_dist_target[range(len(next_actions)), next_actions.squeeze().long()]
        # Replace next_q_dist with a delta distribution at the reward for terminal states
        # done_mask = dones.squeeze().bool()  # convert to bool for indexing
        # next_q_dist[done_mask] = torch.zeros_like(next_q_dist[done_mask])  # set to zeros first
        # reward_indices = 4
        # next_q_dist[done_mask].scatter_(1, reward_indices, 1)
        # print("next_q_dist.size", next_q_dist.sum(-1).sum())
        for t in range(len(dones)):
            if dones[t]:
                next_q_dist[t] = 0.0
                next_q_dist[t][4] = 1.0
        # print("next_q_dist.size",next_q_dist.sum(-1).sum())
        q_dist = q_dist[range(len(actions)), actions.squeeze().long()]
        q_dist.data.clamp_(0.0001, 0.9999)  # to avoid division by 0


        rewards = rewards.expand_as(next_q_dist)
        # print("dones:",dones)
        dones = dones.expand_as(next_q_dist)
        atoms = self.atoms.expand_as(next_q_dist)
        # print("calculate:",(1 - dones) * self.gamma * atoms)
        # Tz = rewards + (1 - dones) * self.gamma * atoms
        # print("Tz:",Tz)
        Tz = 0.0001 + rewards + self.gamma * atoms
        Tz = Tz.clamp(min=self.v_min+0.0001, max=self.v_max-0.0001)
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, self.num_atoms)
        # print("offset.size:",offset.size())
        proj_dist = torch.zeros(next_q_dist.size())
        # print("proj_dist.size:",proj_dist.size())
        # print("next_q_dist.size",next_q_dist.sum(-1).sum())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_q_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_q_dist * (b - l.float())).view(-1))
        # print("proj_dist.size", proj_dist.sum(-1).sum())
        dqn_loss = -(proj_dist * q_dist.log()).sum(1).mean()

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
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
    lr = 1e-4
    epsilon = 0.1
    num_episodes = 20000
    target_update = 5
    buffer_size = 1000
    minimal_size = 500
    batch_size = 128

    hidden_dim = 256
    gamma = 0.9
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    algorithm_name = "DRL"
    env_name = "MUSEUM"
    mode_name = "random"
    robot_num = 1
    target_model = TargetModel(env_name + "_Random")
    env = gym_pqh(env_name, mode_name, robot_num, target_model)
    torch.manual_seed(0)
    state_dim = env.position_embed
    action_dim = env.action_space
    agents = []
    replay_buffers = []

    for i in range(robot_num):
        agent = D_DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
        replay_buffer = ReplayBuffer(buffer_size)
        agents.append(agent)
        replay_buffers.append(replay_buffer)

    return_list = multi_robot_utils_off_policy.train_off_policy_multi_agent(env, agents, replay_buffers, num_episodes,
                                                                            minimal_size, batch_size)

    for h in range(len(agents)):
        env_name = env.env_name
        net_name = "benchmark_save_model/" + env_name + "_DRL_R" + str(len(agents)) + "_R" + str(h)
        torch.save(agents[h].q_net, net_name + '.pth')

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(algorithm_name + ' efficient on {}'.format(env_name) + " ,Num:" + str(robot_num))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 101)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(algorithm_name + ' efficient on {}'.format(env_name) + " ,Num:" + str(robot_num))
    plt.show()