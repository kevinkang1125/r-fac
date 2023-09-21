import math
import numpy as np
import torch
import random
a = torch.tensor([1,2,3])
b = torch.tensor([1,2])
c = torch.zeros([1,3])
c[:,0:len(b)] = b
#c = a + b
print(c)
transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                        'dones': [], 'action_num': []} for _ in range(5)]
print(len(transition_dicts[2]["rewards"]))
counter = 0
rho_list = [2,5,7]
alive_index = [0,1,2]
alive_list = []
while counter < 10:
    if counter in rho_list:
        robot = random.choice(alive_index)
        alive_index.remove(robot)
    print(counter,alive_index)
    alive_list.append(alive_index.copy())
    counter +=1

    print(alive_list)
   


# b = []
# b.append(a)
# b.append(a)
# print(len(a),len(b[0]))
# c = np.exp(a)
# print(c)
# a = torch.tensor(a)
# a = a.pow(2)
# print(a)
# a = range(10)
# print(a)