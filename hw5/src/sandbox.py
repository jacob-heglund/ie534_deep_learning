import sys
import torch
import os
from numpy import shape
import json
from torchvision import models
from collections import OrderedDict
from numpy.linalg import norm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

num_epochs = 13
epoch_arr = np.arange(0, num_epochs, 1)
print(epoch_arr)

