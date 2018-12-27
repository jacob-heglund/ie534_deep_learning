import json
import os
a = [1,2,3,4,5,6]

curr_dir = os.getcwd()
data_fn = 'fake_data.txt'
data_path = os.path.join(curr_dir, data_fn)

with open(data_path, 'w') as f:
    json.dump(a, f)
    