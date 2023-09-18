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
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Environment')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/Quality_Diversity')
sys.path.append('/Users/pqh/PycharmProjects/HandsonRL/Efficient_Search/RL_POMDP')
import rl_utils as rl_utils
from gym_pqh_multi_target import gym_pqh
from Target import TargetModel
import multi_robot_utils_off_policy as multi_robot_utils_off_policy
from ReplayBuffer import ReplayBuffer
import R_Fac_train as rf 

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
        x, _ = self.gru(x)
        x = x * mask.unsqueeze(2).float()
        x = x[torch.arange(x.size(0)), lengths - 1]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        q_value = x.masked_fill(action_mask, float('-inf'))
        return q_value

class Agent:
    def __init__(self,obs_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
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

    def update(self):
        #update target network
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
    
    def opt(self):
        #update q_network
        return self.optimizer
    
    def output_agent(self,transition_dict):
        #wehther to add alive signal
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
        #下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_observations, next_action_masks).max(1)[0].view(
             -1, 1)
        return q_values, max_next_q_values, rewards 

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

class V2DN_pre:
    def __init__(self,agents,gamma_2,agent_num, horizon):
        self.agents = agents
        self.gamma = gamma_2
        self.agent_num = agent_num
        self.horizon = horizon
    def learn(self, alive_index,transition_dicts):
        # Get Q-values for the current state and action for all agents
        joint_current = torch.zeros(self.horizon,1).cuda()
        joint_next = torch.zeros(self.horizon,1).cuda()
        team_reward = torch.zeros(self.horizon,1).cuda()
        current_q_values = torch.zeros(self.horizon,self.agent_num).cuda()
        next_max_q_values = torch.zeros(self.horizon,self.agent_num).cuda()
        rewards = torch.zeros(self.horizon,self.agent_num).cuda()
        for i in alive_index:
            current_q_values[:,i:i+1],next_max_q_values[:,i:i+1],rewards[:,i:i+1] = self.agents[i].output_agent(transition_dicts[i])
            joint_current = joint_current + torch.exp(current_q_values[:,i:i+1])
            joint_next =joint_next + torch.exp(next_max_q_values[:,i:i+1])
            team_reward =team_reward+ rewards[:,i:i+1]
         
        # Compute the joint target Q-value using the team reward
        joint_target_q_value = team_reward + gamma * torch.log(joint_next)
        joint_current_log = torch.log(joint_current)
        
        td_error = joint_target_q_value-joint_current_log
        td_error = torch.sum(td_error.pow(2))+1e-5
        optimizer_list = []
        for i in alive_index:
            opt  = self.agents[i].opt()
            opt.zero_grad()
            optimizer_list.append(opt)
        td_error.backward()
        for m in range(len(optimizer_list)):
            optimizer_list[m].step()
        for a in alive_index:
            self.agents[a].update()
        return td_error

class V2DN_dur:
    def __init__(self,agents,gamma_2,agent_num, horizon):
        self.agents = agents
        self.gamma = gamma_2
        self.agent_num = agent_num
        self.horizon = horizon
    def pre_learn(self, current_q_values, next_max_q_values,rewards, alive_index,transition_dicts):
        # Get Q-values for the current state and action for all agents
        joint_current = torch.zeros(self.horizon,1)
        joint_next = torch.zeros(self.horizon,1)
        team_reward = torch.zeros(self.horizon,1)
        current_q_values = torch.zeros(self.horizon,self.agent_num)
        next_max_q_values = torch.zeros(self.horizon,self.agent_num)
        rewards = torch.zeros(self.horizon,1)
        for i in alive_index:
            current_q_values[:,i:i+1],next_max_q_values[:,i:i+1],rewards[:,i:i+1] = self.agents[i].output_agent(transition_dicts[i])
            joint_current = joint_current + current_q_values[:,i:i+1]
            joint_next =joint_next +next_max_q_values[:,i:i+1]
            team_reward =team_reward+ rewards[:,i:i+1]
         
        # Compute the joint target Q-value using the team reward
        joint_target_q_value = team_reward + gamma * joint_next
        
        td_errors = joint_target_q_value-joint_current
        td_errors = td_errors.detach()
        print(td_errors)
        ####add MSE in the future
        for a in alive_index:
        
            weights = (current_q_values[:,i:i+1]/(joint_current+1e-10))
            individual_td_error = weights*td_errors
            self.agents[a].update(individual_td_error,transition_dicts[a])


            

if __name__ == "__main__":
    lr = 5e-2
    epsilon = 0.15
    num_episodes = 80
    target_update = 2
    iter = 15
     
    rho = 0.9

    hidden_dim = 128
    gamma = 0.95
    gamma_2 = 0.99
    device = torch.device("cuda")
    algo = "V2DN"
    #algo = "VDN"
    

    env_name = "MUSEUM"
    horizon = 70 if env_name =="MUSEUM" else 60
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
        agent = Agent(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
        agents.append(agent)
    
    #mixer = VDN_On_pre(agents,gamma_2,agent_num=robot_num, horizon= horizon)
    mixer = V2DN_pre(agents,gamma_2,agent_num=robot_num, horizon= horizon)

    # return_list = multi_robot_utils_off_policy.train_V2DN_on_policy_multi_agent(env, mixer,agents, replay_buffers, num_episodes,
    #                                                                          batch_size,rho)
    return_list, td_list = rf.train_resilient_on_policy_multi_agent(env, mixer, agents, num_episodes, rho, iter)
    for i in range(robot_num):
        agents[i].save('./on policy robot{} in teamsize{} with rho{} in {}.pth'.format(i,robot_num,rho,env_name))
    episodes_list = list(range(len(return_list)))
    #plt.plot(episodes_list, return_list)
    plt.plot(td_list) 
    plt.xlabel('Episodes')
    plt.ylabel('TD_error')
    plt.title('Performance on {} with rho={} with {}'.format(env_name,rho,algo))
    plt.show()
    np.savetxt("td_list.txt",td_list)
    mv_return = rl_utils.moving_average(td_list, 101)
    plt.plot(mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('average_td')
    plt.title('VPG on {}'.format(env_name))
    plt.show()