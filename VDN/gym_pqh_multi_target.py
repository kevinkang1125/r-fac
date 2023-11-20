# 引入一次多个target,来使robot team训练更合理， 长度统一
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
        self.target_num = 1000
        self.action_space = self.map.map_action_num
        self.embedding_layer = EmbeddingLayer(self.total_position+1, self.position_embed, 0)
        self.mode_name = mode_name
        self.robot_num = robot_num
        self.robot_initial_position = 10 if env_name == "MUSEUM" else 44  
        self.robot_initial_actionNum_set = [self.map.next_total_action(self.robot_initial_position) for _ in range(robot_num)]
        self.robot_position_initial_list = [self.robot_initial_position for _ in range(robot_num)]
        self.target_initial_position = 66 if env_name == "MUSEUM" else 47 
        self.target_random_initial_set = [61, 66, 67, 68, 69] if env_name == "MUSEUM" else [47,48,54,55,59] if env_name == "OFFICE" else None
        self.target_initial_set = [self.target_initial_position for _ in range(self.target_num)]
        self.reward_initial_list = [[] for _ in range(robot_num)]
        self.trajectory_initial_list = [[self.robot_initial_position] for _ in range(robot_num)]
        # *********************************************随机3启动************************************************************：

        # ************************************************************************************************************************
        self.capture_num = 0
        self.target_last_position_set = copy.copy(self.target_initial_set)
        self.target_position_set = copy.copy(self.target_initial_set)
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
        #     self._target_set_move()

        robot_position = self.robot_position_list[robot_label]

        robot_next_position = self.map.step(robot_position, action)

        next_total_action = self.map.next_total_action(robot_next_position)

        # if robot_next_position == self.target_position:
        #     self.done = True
        self._determine_set_capture(robot_position, robot_next_position)

        self.robot_position_list[robot_label] = robot_next_position
        self.trajectory_list[robot_label].append(robot_next_position)
        reward, reward_part2 = self._reward(robot_label)
        self._observation_calculate(robot_label)
        return copy.deepcopy(self.observation_list[robot_label]), copy.deepcopy(self.observation_position3_list[robot_label]), reward, reward_part2, self.done, next_total_action

    def reset(self):
        # print("reward_list", self.)
        print("trajectory_list", self.trajectory_list)
        # print("reward_list",self.reward_list)
        self.capture_num = 0
        self._target_reset(rand=True)
        # self.target_last_position_set = copy.copy(self.target_initial_set)
        self.robot_position_list = copy.copy(self.robot_position_initial_list)
        self.reward_list = copy.deepcopy(self.reward_initial_list)
        # the following setup is only for single_target, used in previous version, this version used target_position_set
        self.target_position = self.target_initial_position
        self.target_last_position = self.target_position
        # ******************************target has 3 initial position************************************
        # self.target_position = random.choice([42,65,8])
        # self.target_last_position = self.target_position
        # ***********************************************************************************************
        self.trajectory_list = copy.deepcopy(self.trajectory_initial_list)
        self._observation_init()
        self.done = False #没有用了
        return copy.deepcopy(self.observation_list), copy.deepcopy(self.observation_position3_list), copy.deepcopy(self.robot_initial_actionNum_set)

    def _target_reset(self,rand = True):

        # In initial place:
        if rand == False:
            self.target_initial_set = [self.target_initial_position for _ in range(self.target_num)]
            self.target_last_position_set = copy.copy(self.target_initial_set)
            self.target_position_set = copy.copy(self.target_initial_set)
        # random start:
        else:
            self.target_initial_set = [random.choice(self.target_random_initial_set) for _ in range(self.target_num)]
            self.target_last_position_set = copy.copy(self.target_initial_set)
            self.target_position_set = copy.copy(self.target_initial_set)

    # reward contains 2 part, capture will get a high reward, return will get a high penalty
    def _reward(self, robot_label):
        reward = 0.0
        robot_position = self.robot_position_list[robot_label]
        target_position = self.target_position
        reward_part1 = -0.1/self.robot_num + 50.0/self.target_num * self.capture_num
        # print("reward_part1:", reward_part1)
        # reward_part2:
        reward_part2 = self._return_penalty(robot_label)
        reward = reward_part1 + reward_part2
        self.reward_list[robot_label].append(reward)
        return reward_part1, reward_part2

    def _observation_init(self):
        for robot_label in range(self.robot_num):
            self.observation_list[robot_label] = self.embedding_layer(torch.tensor(self.trajectory_list[robot_label]))
            self.observation_position3_list[robot_label] = self.embedding_layer(
                torch.tensor(self.trajectory_list[robot_label][-3:] if len(self.trajectory_list[robot_label]) >= 3 else
                             self.trajectory_list[robot_label])) # 3 represents the input to the diversity net

    def _observation_calculate(self, robot_label):
        self.observation_list[robot_label] = self.embedding_layer(torch.tensor(self.trajectory_list[robot_label]))
        self.observation_position3_list[robot_label] = self.embedding_layer(
            torch.tensor(self.trajectory_list[robot_label][-3:] if len(self.trajectory_list[robot_label]) >= 3 else
                         self.trajectory_list[robot_label]))

    def _target_set_move(self):
        self.target_last_position_set = copy.copy(self.target_position_set)
        for i in range(len(self.target_position_set)):
            target_last_position = self.target_position_set[i]
            next_position = self.target_model.next_position(target_last_position)
            self.target_position_set[i] = next_position
        # self.target_last_position = self.target_position
        # print("target_position:",self.target_position)
        # ********************************测试用,一定要删掉************************************
        # self.target_position = self.target_position

        # ***********************************************************************************

    def _determine_set_capture(self, robot_position, robot_next_position):
        self.capture_num = 0
        target_position_set_buffer = copy.copy(self.target_position_set)
        target_last_position_set_buffer = copy.copy(self.target_last_position_set)
        for i in range(len(target_position_set_buffer)):
            target_position = target_position_set_buffer[i]
            target_last_position = target_last_position_set_buffer[i]
            if robot_next_position == target_position:
                a = self.target_position_set.pop(i - self.capture_num)
                b = self.target_last_position_set.pop(i - self.capture_num)
                self.capture_num += 1
            elif robot_position == target_position and robot_next_position == target_last_position:
                a = self.target_position_set.pop(i - self.capture_num)
                b = self.target_last_position_set.pop(i - self.capture_num)
                self.capture_num += 1
        # print("total capture_num in this step: ", self.capture_num, " , length of the target position set:", len(self.target_position_set))

    def _return_penalty(self, robot_label):
        penalty = 0.0
        penalty_base = -1.0
        penalty_weaken = 0.7
        robot_position = self.robot_position_list[robot_label]
        robot_trajectory = self.trajectory_list[robot_label][:-1]
        l = len(robot_trajectory)
        # n = min(10, l)
        for i in range(l):
            if robot_trajectory[l-1-i] == robot_position:
                penalty += penalty_base * (penalty_weaken ** i)
        # **********************************only give reward***********************************
        # robot_trajectory = self.trajectory_list[robot_label][n:-1]
        # if robot_position not in robot_trajectory:
        #     penalty = 1.0
        # **************************************************************************************
        # print("robot_position:", robot_position, "robot_trajectory:", robot_trajectory, "reward_part2:", penalty)
        return penalty
