# 引入一次多个target,来使robot team训练更合理， 长度统一, add epsilon 避免local optimal
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import math
# 两种思路：直接loss_function; 嵌入到reward里

def train_on_policy_multi_agent(env, agents, num_episodes, per_episodes, diveristy_net):
    return_multi_list = []
    epoch_num = 10
    epsilon = 0.1
    for i in range(epoch_num):
        with tqdm(total=int(num_episodes / epoch_num ), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/epoch_num)):
                episode_return = each_epoch_train_on_policy_agent(env, agents, epsilon/(i+1))
                return_multi_list.append(episode_return)
                if i_episode % per_episodes == 0 and i_episode != 0:
                    each_epoch_on_policy_diversity(env, diveristy_net, agents, epsilon/(i+1))
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / epoch_num * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_multi_list[-10:])})
                pbar.update(1)
    return return_multi_list

def train_on_policy_multi_agent_CEPG(env, agents, num_episodes):
    return_multi_list = []
    epoch_num = 10
    epsilon = 0.1
    for i in range(epoch_num):
        with tqdm(total=int(num_episodes / epoch_num ), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/epoch_num)):
                episode_return = each_epoch_train_on_policy_agent_CEPG(env, agents, epsilon/(i+1))
                return_multi_list.append(episode_return)
                # if i_episode % per_episodes == 0 and i_episode != 0:
                #     each_epoch_on_policy_diversity(env, diveristy_net, agents, epsilon/(i+1))
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / epoch_num * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_multi_list[-10:])})
                pbar.update(1)
    return return_multi_list

def train_on_policy_multi_agent_MADDPG(env, agents, num_episodes):
    return_multi_list = []
    epoch_num = 10
    epsilon = 0.1
    for i in range(epoch_num):
        with tqdm(total=int(num_episodes / epoch_num ), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/epoch_num)):
                episode_return = each_epoch_train_on_policy_agent_MADDPG(env, agents, epsilon/(i+1))
                return_multi_list.append(episode_return)
                # if i_episode % per_episodes == 0 and i_episode != 0:
                #     each_epoch_on_policy_diversity(env, diveristy_net, agents, epsilon/(i+1))
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / epoch_num * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_multi_list[-10:])})
                pbar.update(1)
    return return_multi_list

def train_on_policy_multi_agent_MADPG(env, agents, num_episodes):
    return_multi_list = []
    epoch_num = 10
    epsilon = 0.2
    for i in range(epoch_num):
        with tqdm(total=int(num_episodes / epoch_num ), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/epoch_num)):
                episode_return = each_epoch_train_on_policy_agent_MADPG(env, agents, epsilon/(i+1))
                return_multi_list.append(episode_return)
                # if i_episode % per_episodes == 0 and i_episode != 0:
                #     each_epoch_on_policy_diversity(env, diveristy_net, agents, epsilon/(i+1))
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / epoch_num * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_multi_list[-10:])})
                pbar.update(1)
    return return_multi_list

def each_epoch_on_policy_diversity(env, diversity_net, agents, epsilon):
    print("training based on robot diversity")
    episode_return = 0
    num_dicts = len(agents)
    base = math.log(1/num_dicts)
    transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [], 'dones': [],
                         'action_num': []} for _ in range(num_dicts)]
    observations, states, action_nums = env.reset()
    # obs = observations[0]
    # state = states[0]
    team_done = False
    # done = False
    counter = 0
    while counter < 50:
        # print("team_not_done:"+str(counter))
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            if counter == 0:
                obs = observations[i]
                state = states[i]
            else:
                obs = transition_dict['next_observations'][-1]
                state = transition_dict['next_states'][-1]
            action_num = action_nums[i]
            agent = agents[i]
            transition_dict['action_num'].append(action_num)
            action = agent.take_action(obs, action_num, epsilon)
            next_obs, next_state, reward, reward_part2, done, action_num = env.step(action, i)
            action_nums[i] = action_num
            transition_dict['observations'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_observations'].append(next_obs)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)
            # **************************diversity reward*********************************
            # print("obs",obs)
            # print("state:",state)
            probability_distribution = diversity_net.get_probability_distribution(next_state)[0]
            # print("probability_distribution",probability_distribution)
            probability = probability_distribution[i]
            # print("probability_in_utils:",probability)
            reward_part2 = math.log(probability) - base
            # print("reward_part2", reward_part2)
            reward_part2 = max(-3.0, min(reward_part2, 3.0))
            # ***************************************************************************
            transition_dict['rewards'].append(0.0)
            transition_dict['rewards_part2'].append(reward_part2)

            episode_return += reward
            # if done:
            #     team_done = True
        # if team_done:
        #     for i in range(num_dicts):
        #         transition_dicts[i]['dones'][-1] = True
                # transition_dicts[i]['rewards'][-1] += 2.0
        counter += 1
    # if counter == 50:
    #     for i in range(num_dicts):
    #         transition_dicts[i]['dones'][-1] = True
    #         # transition_dicts[i]['rewards'][-1] = -5.0
    #         team_done = True
    for i in range(num_dicts):
        agents[i].update(transition_dicts[i])
    # *****************************diversity network update***************************
    for i in range(num_dicts):
        diversity_net.update(transition_dicts[i], i)
    # ********************************************************************************

def each_epoch_train_on_policy_agent(env, agents, epsilon):
    episode_return = 0.0
    num_dicts = len(agents)
    transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                         'dones': [], 'action_num': []} for _ in range(num_dicts)]
    observations, states, action_nums = env.reset()
    # obs = observations[0]
    team_done = False
    # done = False
    counter = 0
    while counter < 50:
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            if counter == 0:
                obs = observations[i]
            else:
                obs = transition_dict['next_observations'][-1]
            action_num = action_nums[i]
            agent = agents[i]
            action = agent.take_action(obs, action_num, epsilon)
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
        #     if done:
        #         team_done = True
        # if team_done:
        #     for i in range(num_dicts):
        #         transition_dicts[i]['dones'][-1] = True
        counter += 1
    # if counter == 50 and team_done == False:
    #     for i in range(num_dicts):
    #         transition_dicts[i]['dones'][-1] = True
    #         team_done = True

    for i in range(num_dicts):
        # print("transition_dicts["+str(i)+"][dones]", transition_dicts[i]['dones'])
        # print("transition_dicts["+str(i)+"][next_observations]", transition_dicts[i]['next_observations'])
        # print("transition_dicts["+str(i)+"][next_states]", transition_dicts[i]['next_states'])
        agents[i].update(transition_dicts[i])
    return episode_return
    # return len(transition_dicts[0]['rewards'])

def each_epoch_train_on_policy_agent_CEPG(env, agents, epsilon):
    consider_trajectory_cross = False
    episode_return = 0.0
    num_dicts = len(agents)
    transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                         'dones': [], 'action_num': [], 'cross_probs': []} for _ in range(num_dicts)]
    observations, states, action_nums = env.reset()
    # obs = observations[0]
    team_done = False
    # done = False
    counter = 0
    while counter < 60:
        team_reward = 0
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            if counter == 0:
                obs = observations[i]
            else:
                obs = transition_dict['next_observations'][-1]
            action_num = action_nums[i]
            agent = agents[i]
            action = agent.take_action(obs, action_num, epsilon)
            transition_dict['action_num'].append(action_num)
            next_obs, next_state, reward, reward_part2, done, action_num = env.step(action, i)
            action_nums[i] = action_num
            transition_dict['observations'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_observations'].append(next_obs)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            #team_reward += reward
            transition_dict['rewards_part2'].append(reward_part2)
            transition_dict['dones'].append(done)
            episode_return += reward
        # for i in range(num_dicts):
        #     transition_dict = transition_dicts[i]
        #     transition_dict['rewards'].append(team_reward)
        #     if done:
        #         team_done = True
        # if team_done:
        #     for i in range(num_dicts):
        #         transition_dicts[i]['dones'][-1] = True
        counter += 1
    # *********** version not consider trajectory cross ******************:
    if not consider_trajectory_cross:
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            for j in range(num_dicts):
                if j == i:
                    continue
                probs = agents[j].get_probs(transition_dict)
                # probs -= 1.0/num_dicts*(num_dicts-1)
                if len(transition_dict['cross_probs']) == 0:
                    transition_dict['cross_probs'] = probs
                else:
                    transition_dict['cross_probs'] += probs
    # *********** version consider trajectory cross **********************:
    else:
        robot_trajectory_list = env.trajectory_list
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            refer_robot_trajectory = robot_trajectory_list[i]
            for j in range(num_dicts):
                if j == i:
                    continue
                cur_robot_trajectory = robot_trajectory_list[j]
                probs = agents[j].get_probs(transition_dict)
                for i in range(len(probs)):
                    cur_robot_position = cur_robot_trajectory[i]
                    if cur_robot_position not in refer_robot_trajectory[:i+1]:
                        probs[i] = 0.0
                if len(transition_dict['cross_probs']) == 0:
                    transition_dict['cross_probs'] = probs
                else:
                    transition_dict['cross_probs'] += probs
    # if counter == 50 and team_done == False:
    #     for i in range(num_dicts):
    #         transition_dicts[i]['dones'][-1] = True
    #         team_done = True

    for i in range(num_dicts):
        # print("transition_dicts["+str(i)+"][dones]", transition_dicts[i]['dones'])
        # print("transition_dicts["+str(i)+"][next_observations]", transition_dicts[i]['next_observations'])
        # print("transition_dicts["+str(i)+"][next_states]", transition_dicts[i]['next_states'])
        agents[i].update(transition_dicts[i])
    return episode_return

def each_epoch_train_on_policy_agent_MADDPG(env, agents, epsilon):
    consider_trajectory_cross = False
    episode_return = 0.0
    num_dicts = len(agents)
    transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                         'dones': [], 'action_num': [], 'cross_probs': []} for _ in range(num_dicts)]
    observations, states, action_nums = env.reset()
    # obs = observations[0]
    team_done = False
    # done = False
    counter = 0
    while counter < 60:
        team_reward = 0
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            if counter == 0:
                obs = observations[i]
            else:
                obs = transition_dict['next_observations'][-1]
            action_num = action_nums[i]
            agent = agents[i]
            action = agent.take_action(obs, action_num, epsilon)
            transition_dict['action_num'].append(action_num)
            next_obs, next_state, reward, reward_part2, done, action_num = env.step(action, i)
            action_nums[i] = action_num
            transition_dict['observations'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_observations'].append(next_obs)
            transition_dict['next_states'].append(next_state)
            #transition_dict['rewards'].append(reward)
            team_reward += reward
            transition_dict['rewards_part2'].append(reward_part2)
            transition_dict['dones'].append(done)
            episode_return += reward
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            transition_dict['rewards'].append(team_reward)
        #     if done:
        #         team_done = True
        # if team_done:
        #     for i in range(num_dicts):
        #         transition_dicts[i]['dones'][-1] = True
        counter += 1
    # *********** version not consider trajectory cross ******************:
    if not consider_trajectory_cross:
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            for j in range(num_dicts):
                if j == i:
                    continue
                probs = agents[j].get_probs(transition_dict)
                # probs -= 1.0/num_dicts*(num_dicts-1)
                if len(transition_dict['cross_probs']) == 0:
                    transition_dict['cross_probs'] = probs
                else:
                    transition_dict['cross_probs'] += probs
    # *********** version consider trajectory cross **********************:
    else:
        robot_trajectory_list = env.trajectory_list
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            refer_robot_trajectory = robot_trajectory_list[i]
            for j in range(num_dicts):
                if j == i:
                    continue
                cur_robot_trajectory = robot_trajectory_list[j]
                probs = agents[j].get_probs(transition_dict)
                for i in range(len(probs)):
                    cur_robot_position = cur_robot_trajectory[i]
                    if cur_robot_position not in refer_robot_trajectory[:i+1]:
                        probs[i] = 0.0
                if len(transition_dict['cross_probs']) == 0:
                    transition_dict['cross_probs'] = probs
                else:
                    transition_dict['cross_probs'] += probs
    # if counter == 50 and team_done == False:
    #     for i in range(num_dicts):
    #         transition_dicts[i]['dones'][-1] = True
    #         team_done = True

    for i in range(num_dicts):
        # print("transition_dicts["+str(i)+"][dones]", transition_dicts[i]['dones'])
        # print("transition_dicts["+str(i)+"][next_observations]", transition_dicts[i]['next_observations'])
        # print("transition_dicts["+str(i)+"][next_states]", transition_dicts[i]['next_states'])
        agents[i].update(transition_dicts[i])
    return episode_return


def each_epoch_train_on_policy_agent_MADPG(env, agents, epsilon):
    episode_return = 0.0
    num_dicts = len(agents)
    transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                         'dones': [], 'action_num': []} for _ in range(num_dicts)]
    observations, states, action_nums = env.reset()
    # obs = observations[0]
    team_done = False
    # done = False
    counter = 0
    while counter < 50:
        team_reward = 0.0
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            if counter == 0:
                obs = observations[i]
            else:
                obs = transition_dict['next_observations'][-1]
            action_num = action_nums[i]
            agent = agents[i]
            action = agent.take_action(obs, action_num, epsilon)
            transition_dict['action_num'].append(action_num)
            next_obs, next_state, reward, reward_part2, done, action_num = env.step(action, i)
            action_nums[i] = action_num
            transition_dict['observations'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_observations'].append(next_obs)
            transition_dict['next_states'].append(next_state)
            # transition_dict['rewards'].append(reward)
            team_reward += reward
            transition_dict['rewards_part2'].append(reward_part2)
            transition_dict['dones'].append(done)
            episode_return += reward
        for i in range(num_dicts):
            transition_dict = transition_dicts[i]
            transition_dict['rewards'].append(team_reward)
        #     if done:
        #         team_done = True
        # if team_done:
        #     for i in range(num_dicts):
        #         transition_dicts[i]['dones'][-1] = True
        counter += 1
    # if counter == 50 and team_done == False:
    #     for i in range(num_dicts):
    #         transition_dicts[i]['dones'][-1] = True
    #         team_done = True
    print(team_reward)
    for i in range(num_dicts):
        # print("transition_dicts["+str(i)+"][dones]", transition_dicts[i]['dones'])
        # print("transition_dicts["+str(i)+"][next_observations]", transition_dicts[i]['next_observations'])
        # print("transition_dicts["+str(i)+"][next_states]", transition_dicts[i]['next_states'])
        agents[i].update(transition_dicts[i])
    return episode_return