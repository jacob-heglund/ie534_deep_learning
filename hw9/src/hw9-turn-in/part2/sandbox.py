import numpy as np
import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision


a = np.array([[1,2,3], [4,5,6], [7,8,9]])

print(a[0:2, :])