import numpy as np
import os
import sys

trials = [
['adam', '0.01', '30'],
['sgd', '0.1', '60'],
['rmsprop', '0.01', '30'],
]

trial_number = int(sys.argv[1])
opt = trials[trial_number][0]
LR = float(trials[trial_number][1])
number_of_epochs = int(trials[trial_number][2])

print(trial_number)
print(opt)
print(LR)
print(number_of_epochs)
