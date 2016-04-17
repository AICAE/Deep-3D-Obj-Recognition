import os
import numpy as np
it = 0
step_size = 12

x = np.zeros([4260,])
for bla in range(0,4259,12):
    x[bla:bla+12]=bla

# Fisher-Yatest shuffle assuming that rotations of one obj are together
for fy_i in range(4260-1,13,-1 * step_size):
    fy_j = np.random.randint(1,int((fy_i+1)/step_size) + 1) * step_size - 1
    if fy_j - step_size < 0:
        x[fy_i:fy_i-step_size:-1],x[fy_j::-1] = x[fy_j::-1], x[fy_i:fy_i-step_size:-1].copy()
    else:
        x[fy_i:fy_i-step_size:-1], x[fy_j:fy_j-step_size:-1]= x[fy_j:fy_j-step_size:-1], x[fy_i:fy_i-step_size:-1].copy()
