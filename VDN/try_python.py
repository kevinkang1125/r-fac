import math
import numpy as np
import torch
a = torch.tensor([1,2,3])
b = torch.tensor([1,2])
c = torch.zeros([1,3])
c[:,0:len(b)] = b
#c = a + b
print(c)
transition_dicts = [{'observations': [], 'actions': [], 'next_states': [], 'next_observations': [], 'rewards': [], 'rewards_part2': [],
                        'dones': [], 'action_num': []} for _ in range(5)]
print(len(transition_dicts[2]["rewards"]))
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