import torch
import torch.nn as nn
from numpy import shape
import numpy as np


class Net():
    def __init__(self):
        pass
    def add(self, x,y):
        return x+y

net = Net()

a = net.add(1,2)
print(a)