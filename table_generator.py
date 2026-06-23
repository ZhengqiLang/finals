import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

parser = argparse.ArgumentParser()
parser.add_argument('--folders', type=str, required=False,
                    help="Can be either 'main', 'hard', 'sb3' (for stablebaselines), or a manual folder from which to parse all results")
parser.add_argument('--timeout', type=int, required=False, default=1800,
                    help="Timeout (in seconds) that was used for the experiments to parse into the table")
parser.add_argument('--max_allowed_timeouts', type=int, required=False, default=2,
                    help="Above this number of timeouts, the instance will be considered as a timeout overall (and marked as '--' in the table).")

parser = parser.parse_args()

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

SB3_MODE = False
if parser.folders == 'main':
    input_folders = ['main']
elif parser.folders == 'hard':
    input_folders = ['TripleIntegrator', 'PlanarRobot', 'Drone4D']
elif parser.folders == 'sb3':
    input_folders = ['linsys_sb', 'linsys1_sb', 'pendulum_sb', 'collision_sb']
    SB3_MODE = True
else:
    input_folders = [parser.folders]

all_models = ['linear-sys', 'linear-sys (hard layout)', 'pendulum', 'collision-avoid', 'triple-integrator', 'planar-robot', 'drone-4d']

# Get all result files from the specified folders
cwd = os.getcwd()
subfolders = []
for input_folder in input_folders:
    PATH = Path(cwd, 'output', input_folder)
    print('Search path:', PATH)

    if os.path.isdir(PATH):
        folders_found = [f.path for f in os.scandir(PATH) if f.is_dir()]
        subfolders += folders_found
        print(f'- Found {len(folders_found)} experiments to parse')
    else:
        print('- Path does not eixst, so skip')

subfolders.sort()
print(f'Total experiments found: {len(subfolders)}')

if not SB3_MODE:
    dic = {}
    dic_all_prob_bounds = []
    dic_all_cases = ['logRASM+Lip (ours)', 'logRASM', 'Lip', 'baseline']

    for folder in subfolders:
        info_file = Path(folder, 'info.csv')
        args_file = Path(folder, 'args.csv')
        info = pd.read_csv(info_file, index_col=0)['info']
        args = pd.read_csv(args_file, index_col=0)['arguments']

        if info['model'] == 'LinearSystem' and info['layout'] == '0':
            benchmark = 'linear-sys'
        elif info['model'] == 'LinearSystem' and info['layout'] == '1':
            benchmark = 'linear-sys (hard layout)'
        elif info['model'] == 'MyPendulum':
            benchmark = 'pendulum'
        elif info['model'] == 'CollisionAvoidance':
            benchmark = 'collision-avoid'
        elif info['model'] == 'TripleIntegrator':
            benchmark = 'triple-integrator'
        elif info['model'] == 'PlanarRobot':
            benchmark = 'planar-robot'
        elif info['model'] == 'Drone4D':
            benchmark = 'drone-4d'
        else:
            print(f'- Warning: Unknown benchmark model ({info['model']})')
            benchmark = 'unknown'

        if args['weighted'] == 'True' and args['exp_certificate'] == 'True':
            case = dic_all_cases[0]
        elif args['weighted'] == 'False' and args['exp_certificate'] == 'True':
            case = dic_all_cases[1]
        elif args['weighted'] == 'True' and args['exp_certificate'] == 'False':
            case = dic_all_cases[2]
        else:
            case = dic_all_cases[3]

        if benchmark not in dic:
            dic[benchmark] = {}
        if case not in dic[benchmark]:
            dic[benchmark][case] = {}

        p = float(info['probability_bound'])

        # Store probability bound
        if p not in dic_all_prob_bounds:
            dic_all_prob_bounds += [p]

        if p not in dic[benchmark][case]:
            dic[benchmark][case][p] = {
                'runtime': [],
                'success': [],
                'timeouts': 0
            }

        # Parse runtimes
        if info['status'] == 'success' and 'total_CEGIS_time' in info and float(info['total_CEGIS_time']) < parser.timeout:
            dic[benchmark][case][p]['runtime'] += [float(info['total_CEGIS_time'])]
            dic[benchmark][case][p]['success'] += [True if info['status'] == 'success' else False]
        else:
            dic[benchmark][case][p]['runtime'] += [-1]
            dic[benchmark][case][p]['success'] += [False]
            dic[benchmark][case][p]['timeouts'] += 1

    # # Sort probability bounds in ascending order
    dic_all_prob_bounds = np.sort(dic_all_prob_bounds)

    # Convert the parsed data into the desired table format
    columns = ['Benchmark', 'Learner-verifier'] + list(dic_all_prob_bounds)
    DF = pd.DataFrame(columns=columns)

    nBounds = len(dic_all_prob_bounds)
    nCases = len(dic_all_cases)

    latex = [
        '\\begin{tabular}{@{}ll' + ''.join(['l'] * nBounds) + '@{}}',
        '\\toprule',
        '& & \\multicolumn{' + str(nBounds) + '}{c}{{Probability bound $\\rho$}} \\\\',
        '\\cmidrule(lr){3-' + str(3 + nBounds - 1) + '}',
        'Benchmark & Learner-verifier & ' + ' & '.join(list(np.array(dic_all_prob_bounds, dtype=str))) + '\\\\',
    ]

    for model in all_models:
        if model in dic:
            i = 0  # Reset line number for each model
            for case in dic_all_cases:
                if case in dic[model]:
                    # Add line for latex table
                    if i == 0:
                        nCases_current_model = len(dic[model])
                        line = ['\\midrule\\multirow{' + str(nCases_current_model) + '}{*}{\\texttt{' + model + '}}']
                    else:
                        line = ['']
                    line += [' & ' + case]

                    i += 1

                    # Add row to the table
                    row = [model, case]
                    for p in dic_all_prob_bounds:
                        # If entry exists, and the number of timeouts does not exceed the limit
                        # Also check if there is at least success (otherwise, average runtime is negative)
                        if p in dic[model][case] and \
                                dic[model][case][p]['timeouts'] <= parser.max_allowed_timeouts and \
                                any(np.array(dic[model][case][p]['success'], dtype=bool) == True):

                            times = dic[model][case][p]['runtime']
                            time = int(np.round(np.mean(times)))
                            std = int(np.round(np.std(times)))
                            row += [f'{time} (pm {std})']

                            # Add entry to latex table
                            digits = int(np.ceil(np.log10(time)))
                            zeros_to_add = max(0, 3 - digits)
                            phantom = ''.join(['0'] * zeros_to_add)  # Phantoms to improve alignment
                            stars = ''.join(['*'] * dic[model][case][p]['timeouts'])  # Each star represents a timeout
                            line += [' & $\\hphantom{' + phantom + '}$$' + str(time) + ' \\,\\scriptstyle{\\pm ' + str(std) + '}$${}^{' + stars + '}$ ']
                        else:
                            row += ['']
                            line += [' & \\multicolumn{1}{c}{--}']

                    line += [' \\\\']
                    DF.loc[len(DF)] = row

                    latex += [''.join(line)]

    latex += ['\\bottomrule', '\\end{tabular}']

    PATH = Path(cwd, 'output/')
    print('Export table files...')

    if parser.folders == 'main':
        file = Path(PATH, f'main-benchmarks_table_{timestamp}.csv')
        file_tex = Path(PATH, f'main-benchmarks_table_{timestamp}.tex')
    elif parser.folders == 'hard':
        file = Path(PATH, f'hard-benchmarks_table_{timestamp}.csv')
        file_tex = Path(PATH, f'hard-benchmarks_table_{timestamp}.tex')
    else:
        file = Path(PATH, f'runtimes_table_{timestamp}.csv')
        file_tex = Path(PATH, f'runtimes_table_{timestamp}.tex')

    # Export Pandas table
    print(f'- Export Pandas DataFrame to: {file}')
    DF.to_csv(file, index=False)

    # Export Latex table
    print(f'- Export LaTeX table to: {file_tex}')
    lines = map(lambda x: x + '\n', latex)
    f = open(file_tex, 'w')
    f.writelines(lines)
    f.close()

    ######

if SB3_MODE:
    dic_sb = {}
    dic_all_steps = []
    dic_all_algos = []
    dic_all_settings = []

    for folder in subfolders:
        info_file = Path(folder, 'info.csv')
        args_file = Path(folder, 'args.csv')
        info = pd.read_csv(info_file, index_col=0)['info']
        args = pd.read_csv(args_file, index_col=0)['arguments']

        if info['model'] == 'LinearSystem' and info['layout'] == '0':
            benchmark = 'linear-sys'
        elif info['model'] == 'LinearSystem' and info['layout'] == '1':
            benchmark = 'linear-sys (hard layout)'
        elif info['model'] == 'MyPendulum':
            benchmark = 'pendulum'
        elif info['model'] == 'CollisionAvoidance':
            benchmark = 'collision-avoid'
        elif info['model'] == 'TripleIntegrator':
            benchmark = 'triple-integrator'
        elif info['model'] == 'PlanarRobot':
            benchmark = 'planar-robot'
        elif info['model'] == 'Drone4D':
            benchmark = 'drone-4d'
        else:
            print(f'- Warning: Unknown benchmark model ({info['model']})')
            benchmark = 'unknown'

        steps = int(info['ckpt'].split('steps=')[1])
        algo = info['ckpt'].split('alg=')[1].split('_')[0]
        setting = r"\alpha=" + f'{float(args.expDecr_multiplier)}' + r", \tau=" + f'{float(args.mesh_loss)}'

        if steps not in dic_all_steps:
            dic_all_steps += [steps]
        if algo not in dic_all_algos:
            dic_all_algos += [algo]
        if setting not in dic_all_settings:
            dic_all_settings += [setting]

        if benchmark not in dic_sb:
            dic_sb[benchmark] = {}
        if steps not in dic_sb[benchmark]:
            dic_sb[benchmark][steps] = {}
        if setting not in dic_sb[benchmark][steps]:
            dic_sb[benchmark][steps][setting] = {}
        if algo not in dic_sb[benchmark][steps][setting]:
            dic_sb[benchmark][steps][setting][algo] = {
                'runtime': [],
                'success': [],
                'timeouts': 0
            }

        # Parse runtimes
        if info['status'] == 'success' and 'total_CEGIS_time' in info and float(info['total_CEGIS_time']) < parser.timeout:
            dic_sb[benchmark][steps][setting][algo]['runtime'] += [float(info['total_CEGIS_time'])]
            dic_sb[benchmark][steps][setting][algo]['success'] += [True if info['status'] == 'success' else False]
        else:
            dic_sb[benchmark][steps][setting][algo]['runtime'] += [-1]
            dic_sb[benchmark][steps][setting][algo]['success'] += [False]
            dic_sb[benchmark][steps][setting][algo]['timeouts'] += 1

    # Sort probability bounds in ascending order
    dic_all_steps = np.sort(dic_all_steps)
    dic_all_algos = list(np.sort(dic_all_algos)[::-1])

    # Convert the parsed data into the desired table format
    columns = ['Benchmark', 'Steps']
    for setting in dic_all_settings:
        columns += [algo + '(' + setting + ')' for algo in dic_all_algos]

    DF = pd.DataFrame(columns=columns)

    nAlgo = len(dic_all_algos)
    nSettings = len(dic_all_settings)
    nSteps = len(dic_all_steps)

    latex = [
        '\\begin{tabular}{@{}ll' + ''.join(['l'] * nAlgo * nSettings) + '@{}}',
        '\\toprule',
        '& & ' + ' '.join(['\\multicolumn{' + str(nAlgo) + '}{c}{{$' + setting + '$}}' for setting in dic_all_settings]) + '\\\\',
        ' '.join(['\\cmidrule(lr){' + str(3 + i * nAlgo) + '-' + str(3 + (i + 1) * nAlgo - 1) + '}' for i, setting in enumerate(dic_all_settings)]),
        'Benchmark & Steps & ' + ' & '.join(dic_all_algos * nSettings) + '\\\\',
    ]

    for model in all_models:
        if model in dic_sb:
            i = 0  # Reset line number for each model
            for steps in dic_all_steps:
                if steps in dic_sb[model]:
                    # Add line for latex table
                    if i == 0:
                        nSteps_current_model = len(dic_sb[model])
                        line = ['\\midrule\\multirow{' + str(nSteps_current_model) + '}{*}{\\texttt{' + model + '}}']
                    else:
                        line = ['']
                    line += [' & $\\num{' + '{:.0e}'.format(steps) + '}$']

                    i += 1

                    # Add row to the table
                    row = [model, steps]
                    for setting in dic_all_settings:
                        for algo in dic_all_algos:
                            # If entry exists, and the number of timeouts does not exceed the limit
                            # Also check if there is at least success (otherwise, average runtime is negative)
                            if setting in dic_sb[model][steps] and algo in dic_sb[model][steps][setting] and \
                                    dic_sb[model][steps][setting][algo]['timeouts'] <= parser.max_allowed_timeouts and \
                                    any(np.array(dic_sb[model][steps][setting][algo]['success'], dtype=bool) == True):

                                times = dic_sb[model][steps][setting][algo]['runtime']
                                time = int(np.round(np.mean(times)))
                                std = int(np.round(np.std(times)))
                                row += [f'{time} (pm {std})']

                                # Add entry to latex table
                                digits = int(np.ceil(np.log10(time)))
                                zeros_to_add = max(0, 3 - digits)
                                phantom = ''.join(['0'] * zeros_to_add)  # Phantoms to improve alignment
                                stars = ''.join(['*'] * dic_sb[model][steps][setting][algo]['timeouts'])  # Each star represents a timeout
                                line += [' & $\\hphantom{' + phantom + '}$$' + str(time) + ' \\,\\scriptstyle{\\pm ' + str(std) + '}$${}^{' + stars + '}$ ']
                            else:
                                row += ['']
                                line += [' & \\multicolumn{1}{c}{--}']

                    line += [' \\\\']
                    DF.loc[len(DF)] = row

                    latex += [''.join(line)]

    latex += ['\\bottomrule', '\\end{tabular}']

    PATH = Path(cwd, 'output/')
    print('Export table files...')

    # Export Pandas table
    file = Path(PATH, f'SB3-benchmarks_{timestamp}.csv')
    print(f'- Export Pandas DataFrame to: {file}')
    DF.to_csv(file, index=False)

    # Export Latex table
    file = Path(PATH, f'SB3-benchmarks_{timestamp}.tex')
    print(f'- Export LaTeX table to: {file}')
    lines = map(lambda x: x + '\n', latex)
    f = open(file, 'w')
    f.writelines(lines)
    f.close()
