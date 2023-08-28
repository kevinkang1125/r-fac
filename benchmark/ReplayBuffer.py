# replay buffer used for futher training:
import collections
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, observation, action, action_num, reward, reward_part2, next_observation, next_action_num, done):
        self.buffer.append((observation, action, action_num, reward, reward_part2, next_observation, next_action_num,
                            done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        observation, action, action_num, reward, reward_part2, next_observation, next_action_num, done = \
            zip(*transitions)
        return observation, action, action_num, reward, reward_part2, next_observation,next_action_num, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)