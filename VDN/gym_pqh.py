from Map import Map
from Target import TargetModel
from Embedding import EmbeddingLayer
import torch
import copy
import random
class gym_pqh:
    def __init__(self, env_name, mode_name, robot_num, target_model):
        self.env_name = env_name
        self.map = Map(env_name)
        self.target_model = target_model
        self.total_position = self.map.map_position_num
        self.position_embed = 5
        self.action_space = self.map.map_action_num
        self.embedding_layer = EmbeddingLayer(self.total_position+1, self.position_embed, 0)
        self.mode_num = mode_name
        self.robot_num = robot_num
        self.robot_initial_position = 40 if env_name == "MUSEUM" else 43 if env_name == "OFFICE" else None
        self.robot_initial_actionNum_set = [self.map.next_total_action(self.robot_initial_position) for _ in range(robot_num)]
        self.robot_position_initial_list = [self.robot_initial_position for _ in range(robot_num)]
        self.target_initial_position = 66 if env_name == "MUSEUM" else 48 if env_name == "OFFICE" else None
        # *********************************************随机3启动************************************************************：
        self.target_random_initial_set = [61, 66, 67, 68, 69] if env_name == "MUSEUM" else [47,48,54,55,59] if env_name == "OFFICE" else None
        # ************************************************************************************************************************
        self.trajectory_initial_list = [[self.robot_initial_position] for _ in range(robot_num)]
        self.reward_initial_list = [[] for _ in range(robot_num)]
        # self.observation_initial_list = [[self.robot_initial_position] for _ in range(robot_num)]
        # self.observation_position3_initial_list = [[self.robot_initial_position] for _ in range(robot_num)]

        self.reward_list = copy.deepcopy(self.reward_initial_list)
        self.robot_position_list = copy.copy(self.robot_position_initial_list)
        self.observation_list = [[] for _ in range(robot_num)]
        self.observation_position3_list = [[] for _ in range(robot_num)]
        self.target_position = self.target_initial_position
        self.target_last_position = self.target_position
        self.trajectory_list = copy.deepcopy(self.trajectory_initial_list)
        self._observation_init()
        self.done = False


    def setup(self, robot_initial_position_list, target_initial_position, target_model):
        self.robot_position_initial_list = robot_initial_position_list
        self.target_initial_position = target_initial_position
        self.target_model = target_model

    def step(self, action, robot_label):
        # if robot_label == 0:
        #     self._target_move()
        robot_position = self.robot_position_list[robot_label]

        robot_next_position = self.map.step(robot_position,action)

        next_total_action = self.map.next_total_action(robot_next_position)

        # if robot_next_position == self.target_position:
        #     self.done = True
        self._determine_capture(robot_position, robot_next_position)

        self.robot_position_list[robot_label] = robot_next_position
        self.trajectory_list[robot_label].append(robot_next_position)
        reward = self._reward(robot_label)
        self._observation_calculate(robot_label)
        return copy.deepcopy(self.observation_list[robot_label]), copy.deepcopy(self.observation_position3_list[robot_label]), reward, self.done, next_total_action

    def reset(self):
        # print("reward_list", self.)
       # print("trajectory_list", self.trajectory_list)
        # print("reward_list",self.reward_list)
        self.robot_position_list = copy.copy(self.robot_position_initial_list)
        self.reward_list = copy.deepcopy(self.reward_initial_list)
        self.target_position = self.target_initial_position
        self.target_last_position = self.target_position
        # ******************************target has 3 initial position************************************
        # self.target_position = random.choice([42,65,8])
        # self.target_last_position = self.target_position
        # ***********************************************************************************************
        self.trajectory_list = copy.deepcopy(self.trajectory_initial_list)
        self._observation_init()
        self.done = False
        return copy.deepcopy(self.observation_list), copy.deepcopy(self.observation_position3_list), copy.deepcopy(self.robot_initial_actionNum_set)

    # reward contains 2 part, capture will get a high reward, return will get a high penalty
    def _reward(self, robot_label):
        reward = 0.0
        reward_part1 = -0.1
        robot_position = self.robot_position_list[robot_label]
        target_position = self.target_position
        if self.done:
            reward_part1 = 10.0
        # reward_part2:
        reward_part2 = self._return_penalty(robot_label)
        reward = reward_part1 + reward_part2
        self.reward_list[robot_label].append(reward)
        return reward

    def _observation_init(self):
        for robot_label in range(self.robot_num):
            self.observation_list[robot_label] = self.embedding_layer(torch.tensor(self.trajectory_list[robot_label]))
            self.observation_position3_list[robot_label] = self.embedding_layer(
                torch.tensor(self.trajectory_list[robot_label][-2:] if len(self.trajectory_list[robot_label]) >= 2 else
                             self.trajectory_list[robot_label]))

    def _observation_calculate(self, robot_label):
        self.observation_list[robot_label] = self.embedding_layer(torch.tensor(self.trajectory_list[robot_label]))
        self.observation_position3_list[robot_label] = self.embedding_layer(
            torch.tensor(self.trajectory_list[robot_label][-2:] if len(self.trajectory_list[robot_label]) >= 2 else
                         self.trajectory_list[robot_label]))

    def _target_move(self):
        self.target_last_position = self.target_position
        next_position = self.target_model.next_position(self.target_position)
        self.target_position = next_position
        # print("target_position:",self.target_position)
        # ********************************测试用,一定要删掉************************************
        # self.target_position = self.target_position

        # ***********************************************************************************

    def _determine_capture(self,robot_position, robot_next_position):
        self.done = False
        if robot_next_position == self.target_position:
            self.done = True
        if robot_position == self.target_position and robot_next_position == self.target_last_position:
            self.done = True

    def _return_penalty(self, robot_label):
        penalty = 0.0
        penalty_base = -1.0
        penalty_weaken = 0.8
        robot_position = self.robot_position_list[robot_label]
        robot_trajectory = self.trajectory_list[robot_label][:-1]
        l = len(robot_trajectory)
        n = min(10, l)
        for i in range(n):
            if robot_trajectory[l-1-i] == robot_position:
                penalty += penalty_base * (penalty_weaken ** i)
        # **********************************only give reward***********************************
        # robot_trajectory = self.trajectory_list[robot_label][n:-1]
        # if robot_position not in robot_trajectory:
        #     penalty = 1.0
        # **************************************************************************************
        return penalty
