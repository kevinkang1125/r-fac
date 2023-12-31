# the first version off-policy training utils
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import math

def train_off_policy_multi_agent(env, agents, replay_buffers, num_episodes, minimal_size, batch_size):
    return_multi_list = []
    epoch_num = 10
    for i in range(epoch_num):
        with tqdm(total=int(num_episodes / epoch_num), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / epoch_num)):
                episode_return = each_sampling(env, agents, replay_buffers)
                return_multi_list.append(episode_return)
                if replay_buffers[0].size() > minimal_size:
                    each_train_off_policy(agents, replay_buffers,batch_size)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / epoch_num * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_multi_list[-10:])})
                pbar.update(1)
    return return_multi_list

def train_off_policy_multi_agent_VDN(env,mixer,agents, replay_buffers, num_episodes, minimal_size, batch_size):
    return_multi_list = []
    epoch_num = 10
    for i in range(epoch_num):
        with tqdm(total=int(num_episodes / epoch_num), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / epoch_num)):
                episode_return = each_sampling(env, agents, replay_buffers)
                return_multi_list.append(episode_return)
                if replay_buffers[0].size() > minimal_size:
                    central_train_off_policy(mixer,replay_buffers,batch_size,agents)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / epoch_num * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_multi_list[-10:])})
                pbar.update(1)
    return return_multi_list

def each_sampling(env, agents, replay_buffers):
    episode_return = 0.0
    num_dicts = len(agents)
    observations, states, action_nums = env.reset()
    horizon = 60 if env.env_name == "MUSEUM" else 70
    counter = 0
    while counter < horizon:
        obs_list = env.observation_list
        for i in range(num_dicts):
            agent = agents[i]
            replay_buffer = replay_buffers[i]
            if counter == 0:
                obs = observations[i]
            else:
                obs = obs_list[i]
            action_num = action_nums[i]
            action = agent.take_action(obs, action_num)
            next_obs, next_state, reward, reward_part2, done, next_action_num = env.step(action, i)
            action_nums[i] = next_action_num
            if counter == 49:
                done = True
            replay_buffer.add(obs, action, action_num, reward, reward_part2, next_obs,
                              next_action_num, done)
            episode_return += reward
        counter += 1
    return episode_return

def each_train_off_policy(agents, replay_buffers,batch_size):
    for i in range(len(agents)):
        agent = agents[i]
        replay_buffer = replay_buffers[i]
        b_o, b_a, b_an, b_r, b_r2, b_no, b_nan, b_d = replay_buffer.sample(batch_size)
        transition_dict = {
            'observations': b_o,
            'actions': b_a,
            'action_num': b_an,
            'rewards': b_r,
            'rewards_part2': b_r2,
            'next_observations': b_no,
            'next_action_num': b_nan,
            'dones': b_d
        }
        agent.update(transition_dict)

def central_train_off_policy(mixer,replay_buffers,batch_size,agents):

    transition_dicts = []
    for i in range(len(agents)):
        agent = agents[i]
        replay_buffer = replay_buffers[i]
        b_o, b_a, b_an, b_r, b_r2, b_no, b_nan, b_d = replay_buffer.sample(batch_size)
        transition_dict = {
            'observations': b_o,
            'actions': b_a,
            'action_num': b_an,
            'rewards': b_r,
            'rewards_part2': b_r2,
            'next_observations': b_no,
            'next_action_num': b_nan,
            'dones': b_d
        }
        transition_dicts.append(transition_dict)
    #print(len(transition_dicts['actions']))
   
    mixer.learn(agents,transition_dicts)

