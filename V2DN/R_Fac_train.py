from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import math


def train_resilient_on_policy_multi_agent(env, mixer, agents, num_episodes, rho,iter):
    return_multi_list = []
    td_list = []
    epoch_num = 10
    #epsilon = 0.1
    for i in range(epoch_num):
        with tqdm(total=int(num_episodes / epoch_num), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / epoch_num)):
                episode_return,alive_index,trainsition_dicts = faulty_sampling_pre(env, agents,rho)
                return_multi_list.append(episode_return)
                td_error = central_train_on_policy_pre(mixer, trainsition_dicts,alive_index,iter,agents)
                td_list.extend(td_error)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / epoch_num * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_multi_list[-10:])})
                pbar.update(1)
    return return_multi_list,td_list

def train_resilient_on_policy_multi_agent_dur(env, mixer, agents, num_episodes,rho_list,iter):
    return_multi_list = []
    td_list = []
    epoch_num = 10
    #epsilon = 0.1
    for i in range(epoch_num):
        with tqdm(total=int(num_episodes / epoch_num), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / epoch_num)):
                episode_return,alive_lists,trainsition_dicts = faulty_sampling_during(env, agents,rho_list)
                return_multi_list.append(episode_return)
                td_error = central_train_on_policy_dur(mixer, trainsition_dicts,alive_lists,iter,agents)
                td_list.extend(td_error)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / epoch_num * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_multi_list[-10:])})
                pbar.update(1)
    return return_multi_list,td_list

def faulty_sampling_pre(env, agents, rho):
    episode_return = 0.0
    agent_num = len(agents)
    transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                         'dones': [], 'action_num': []} for _ in range(agent_num)]
    alive_list = []
    ##faulty sample
    for i in range(agent_num):
        if np.random.random() < rho:
            alive_list.append(i)
    horizon = 70 if env.env_name =="MUSEUM" else 60
    ##change into index
    num_dicts = alive_list
    observations, states, action_nums = env.reset()
    counter = 0
    if len(num_dicts)== 0:
        num_dicts = [0]
    factor = agent_num/len(num_dicts)
    while counter < horizon:
        # obs_list = env.observation_list
        env._target_set_move()
        for i in (num_dicts):
            transition_dict = transition_dicts[i]
            if counter == 0:
                obs = observations[i]
            else:
                obs = transition_dict['next_observations'][-1]
            agent = agents[i]
            action_num = action_nums[i]
            action = agent.take_action(obs, action_num)
            transition_dict['action_num'].append(action_num)
            next_obs, next_state, reward, reward_part2, done, action_num = env.step(action, i)
            action_nums[i] = action_num
            transition_dict['observations'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_observations'].append(next_obs)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['rewards_part2'].append(reward_part2)
            transition_dict['dones'].append(done)
            episode_return += reward
        counter += 1
    return episode_return*factor,alive_list,transition_dicts

def faulty_sampling_during(env, agents, rho_list):
    episode_return = 0.0
    agent_num = len(agents)
    transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                         'dones': [], 'action_num': []} for _ in range(agent_num)]
    alive_lists = []
    alive_index = []
    for m in range(agent_num):
        alive_index.append(m)
    observations, states, action_nums = env.reset()
    horizon = 70 if env.env_name =="MUSEUM" else 60
    counter = 0
    while counter < horizon:
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
            next_obs, next_state, reward, reward_part2, done, action_num = env.step(action, i)
            action_nums[i] = action_num
            transition_dict['observations'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_observations'].append(next_obs)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['rewards_part2'].append(reward_part2)
            transition_dict['dones'].append(done)
            episode_return += reward
        counter += 1
        alive_lists.append(alive_index.copy())
    return episode_return,alive_lists,transition_dicts

def central_train_on_policy_pre(mixer,transition_dicts,alive_index,iter,agents):
    td_list = []
    for m in range (iter):
        td_error = mixer.learn(alive_index,transition_dicts,agents)
        td_list.append(td_error.item())
    return td_list

def central_train_on_policy_dur(mixer,transition_dicts,alive_list,iter,agents):
    td_list = []
    if len(alive_list) <= 10:
        print("Wrong Usage of During Execution Mixer")
    else:
        for m in range (iter):
            td_error = mixer.learn(alive_list,transition_dicts,agents)
            td_list.append(td_error.item())
    return td_list
      
    