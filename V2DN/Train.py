import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence







class V2DN_pre:
    def __init__(self,gamma_2,agent_num, horizon):
        #self.agents = agents
        self.gamma = gamma_2
        self.agent_num = agent_num
        self.horizon = horizon
    def learn(self, alive_index,transition_dicts,agents):
        # Get Q-values for the current state and action for all agents
        joint_current = torch.zeros(self.horizon,requires_grad=True).cuda()
        joint_next = torch.zeros(self.horizon,1,requires_grad=True).cuda()
        team_reward = torch.zeros(self.horizon,1).cuda()
        current_q_values = torch.zeros(self.horizon,self.agent_num,requires_grad=True).cuda()
        next_max_q_values = torch.zeros(self.horizon,self.agent_num,requires_grad=True).cuda()
        rewards = torch.zeros(self.horizon,self.agent_num,requires_grad=True).cuda()
        for i in alive_index:
            current_q_values[:,i:i+1],next_max_q_values[:,i:i+1],rewards[:,i:i+1] = agents[i].output_agent(transition_dicts[i])
            joint_current = joint_current + torch.exp(current_q_values[:,i:i+1])
            joint_next =joint_next + torch.exp(next_max_q_values[:,i:i+1])
            team_reward =team_reward+ rewards[:,i:i+1]
         
        # Compute the joint target Q-value using the team reward
        joint_target_q_value = team_reward + self.gamma * torch.log(joint_next)
        joint_current_log = torch.log(joint_current)
        
        td_error = joint_target_q_value-joint_current_log
        td_error = torch.sum(td_error.pow(2))+1e-5
        optimizer_list = []
        for i in alive_index:
            opt  = agents[i].opt()
            opt.zero_grad()
            optimizer_list.append(opt)
        #td_error.requires_grad = True
        td_error.backward()
        for m in range(len(optimizer_list)):
            optimizer_list[m].step()
        for a in alive_index:
            agents[a].update()
        return td_error

class V2DN_dur:
    def __init__(self,gamma_2,agent_num, horizon):
        #self.agents = agents
        self.gamma = gamma_2
        self.agent_num = agent_num
        self.horizon = horizon
    def learn(self, alive_list,transition_dicts,agents):
        # Get Q-values for the current state and action for all agents
        joint_current = torch.zeros(self.horizon,requires_grad=True).cuda()
        joint_next = torch.zeros(self.horizon,1,requires_grad=True).cuda()
        team_reward = torch.zeros(self.horizon,1).cuda()
        current_q_values = torch.zeros(self.horizon,self.agent_num,requires_grad=True).cuda()
        next_max_q_values = torch.zeros(self.horizon,self.agent_num,requires_grad=True).cuda()
        rewards = torch.zeros(self.horizon,self.agent_num,requires_grad=True).cuda()
        for i in range(self.agent_num):
            ep_len = len(transition_dicts[i]["rewards"])
            current_q_values[0:ep_len,i:i+1],next_max_q_values[0:ep_len,i:i+1],rewards[0:ep_len,i:i+1] = agents[i].output_agent(transition_dicts[i])
            joint_current[0:ep_len] = joint_current[0:ep_len] + torch.exp(current_q_values[0:ep_len,i:i+1].squeeze())
            joint_next[0:ep_len] =joint_next[0:ep_len] + torch.exp(next_max_q_values[0:ep_len,i:i+1])
            team_reward[0:ep_len] =team_reward[0:ep_len]+ rewards[0:ep_len,i:i+1]
         
        # Compute the joint target Q-value using the team reward
        joint_target_q_value = team_reward + self.gamma * torch.log(joint_next)
        joint_current_log = torch.log(joint_current)
        
        td_error = joint_target_q_value-joint_current_log
        td_error = torch.sum(td_error.pow(2))+1e-5
        optimizer_list = []
        for i in range(self.agent_num):
            opt  = agents[i].opt()
            opt.zero_grad()
            optimizer_list.append(opt)
        td_error.backward()
        for m in range(len(optimizer_list)):
            optimizer_list[m].step()
        for a in range(self.agent_num):
            agents[a].update()
        return td_error