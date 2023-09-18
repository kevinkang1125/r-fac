import math
import numpy as np
import torch
a = [1,2,3]
b = []
b.append(a)
b.append(a)
print(len(a),len(b[0]))
c = np.exp(a)
print(c)
a = torch.tensor(a)
a = a.pow(2)
print(a)