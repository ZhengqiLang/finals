import os
from pathlib import Path

import numpy as np
import pandas as pd

# Parameters
input_folder = 'output/main/'
cwd = os.getcwd()

PATH = Path(cwd, input_folder)
print('Path:', PATH)

subfolders = [f.path for f in os.scandir(PATH) if f.is_dir()]
subfolders.sort()
validate_ckpts = []
DIC = {}
print(f'- Found {len(subfolders)} experiment instances to parse')

# Iterate a first time over all folder to extract the probability bounds
probability_bounds = []

dic = {}

for folder in subfolders:
    info_file = Path(folder, 'info.csv')
    args_file = Path(folder, 'args.csv')
    info = pd.read_csv(info_file, index_col=0)['info']
    args = pd.read_csv(args_file, index_col=0)['arguments']

    if info['seed'] not in ['1', '2', '3']:
        continue

    if info['model'] == 'LinearSystem' and info['layout'] == '0':
        benchmark = 0
    elif info['model'] == 'LinearSystem' and info['layout'] == '1':
        benchmark = 1
    elif info['model'] == 'MyPendulum':
        benchmark = 2
    else:
        benchmark = 3

    if benchmark not in dic:
        dic[benchmark] = {}

    if args['weighted'] == 'True' and args['exp_certificate'] == 'True':
        case = 'ours'
    elif args['weighted'] == 'False' and args['exp_certificate'] == 'True':
        case = 'no-lip'
    elif args['weighted'] == 'True' and args['exp_certificate'] == 'False':
        case = 'no-exp'
    else:
        case = 'base'

    if case not in dic[benchmark]:
        dic[benchmark][case] = {}

    p = float(info['probability_bound'])
    if p not in dic[benchmark][case]:
        dic[benchmark][case][p] = 0

    if info['status'] == 'success' and 'total_CEGIS_time' in info:
        dic[benchmark][case][p] += 1

# Now generate shell script to run remaining experiments
MAIN = [
    '#!/bin/bash',
    '',
    'verify_batch_size=30000',
    'forward_pass_batch_size=1000000',
    'time_mul=1',
    'if [ ! -z "$1" ]; then verify_batch_size=$1; fi',
    'if [ ! -z "$2" ]; then forward_pass_batch_size=$2; fi',
    'if [ ! -z "$3" ]; then time_mul=$3; fi',
    'collision_batch_size=$((verify_batch_size/4))',
    '',
    'models=("LinearSystem --verify_batch_size ${verify_batch_size}" "LinearSystem --layout 1 --verify_batch_size ${verify_batch_size}" "MyPendulum --verify_batch_size ${verify_batch_size}" "CollisionAvoidance --noise_partition_cells 24 --verify_batch_size ${collision_batch_size}"))',
    'all_flags="--logger_prefix main --eps_decrease 0.01 --ppo_max_policy_lipschitz 10 --hidden_layers 3 --expDecr_multiplier 10 --pretrain_method PPO_JAX --pretrain_total_steps 100000 --refine_threshold 250000000 --forward_pass_batch_size ${forward_pass_batch_size}"',
    'flags_mesh1="--mesh_loss 0.001 --mesh_loss_decrease_per_iter 0.8"',
    'flags_mesh2="--mesh_loss 0.01 --mesh_loss_decrease_per_iter 0.8"',
    '',
    'TO=$((1800*time_mul+200)) # Add 200 seconds to avoid that pretraining causes a timeout',
    'TOtable=$((1800*time_mul))',
    '',
]

X = ['Linsys layout=0', 'Linsys layout=1', 'Pendulum', 'CollisionAvoidance']
cases = ['ours', 'no-lip', 'no-exp', 'base']

MAIN += ['for seed in {4..10};', 'do']
for i in range(4):
    MAIN += [f'  # {X[i]}']
    for case in cases:
        MAIN += [f'  # {case}']

        current = dic[i][case]
        probabilities = np.sort(list(current.keys()))

        if i < 3:
            mesh = '1'
        else:
            mesh = '2'

        if np.any(np.array(list(current.values())) > 0):
            line = ' '.join(map(str, [f'{p}' for p in probabilities if current[p] > 0]))
            MAIN += [f'  for p in {line}', '  do']
            if case == 'ours':
                MAIN += ['    timeout $TO python run.py --seed $seed --model ${models[' + str(i) + ']} $all_flags $flags_mesh' + str(
                    mesh) + ' --probability_bound $p --exp_certificate;']
            elif case == 'no-lip':
                MAIN += ['    timeout $TO python run.py --seed $seed --model ${models[' + str(i) + ']} $all_flags $flags_mesh' + str(
                    mesh) + ' --probability_bound $p --exp_certificate --no-weighted --no-cplip;']
            elif case == 'no-exp':
                MAIN += ['    timeout $TO python run.py --seed $seed --model ${models[' + str(i) + ']} $all_flags $flags_mesh' + str(
                    mesh) + ' --probability_bound $p --no-exp_certificate;']
            else:
                MAIN += ['    timeout $TO python run.py --seed $seed --model ${models[' + str(i) + ']} $all_flags $flags_mesh' + str(
                    mesh) + ' --probability_bound $p --no-exp_certificate --no-weighted --no-cplip;']
            MAIN += ['  done']
    MAIN += ['  ']
MAIN += ['done']

PATH = Path(cwd, 'experiments/')
print('Export shell script to folder:', PATH)

lines = map(lambda x: x + '\n', MAIN)
f = open(Path(PATH, 'run_main_otherSeeds.sh'), 'w')
f.writelines(lines)
f.close()
