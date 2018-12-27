import numpy as np
import os
a = np.zeros([3, 10])

a[2, 5] = 5
x = 4
for i in range(10):
    if x > a[2, i]:
        a[2, i] = x

print(a)

