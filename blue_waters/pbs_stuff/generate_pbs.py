# This script creates a PBS file that runs one hyperameter setting
# on a single node.
import os

# generates a file called 'run.pbs' in the directory this python file is in
pbs_fn = 'run.pbs'
curr_dir = os.getcwd()
pbs_path = os.path.join(curr_dir, pbs_fn)

# Change this if your hyperparameters change
trial_number = 0

trainingFilename = 'test_job.py'
walltime = '00:30:00'
jobname ='my_training_job_name'
netid = 'jheglun2'
directory = curr_dir.replace('\\', '/')

with open(pbs_path, 'w') as f:
    f.write("#!/bin/bash\n")
    
    f.write("#PBS -l nodes=1:ppn=16:xk\n")
    f.write("#PBS -N {0}_{1}\n".format(jobname,trial_number))
    f.write("#PBS -l walltime={0}\n".format(walltime))
    f.write("#PBS -e $PBS_JOBNAME.$PBS_JOBID.err\n")
    f.write("#PBS -o $PBS_JOBNAME.$PBS_JOBID.out\n")
    f.write("#PBS -M {0}@illinois.edu\n".format(netid))
    f.write("#PBS -m bea\n")
    
    f.write("cd {0}\n".format(directory))
    f.write(". /opt/modules/default/init/bash\n")
    f.write('module load bwpy/2.0.0-pre1\n')
    f.write('module load cudatoolkit\n')
    f.write("aprun -n 1 -N 1 python3.6 {0} {1}\n".format(trainingFilename, trial_number))
    
