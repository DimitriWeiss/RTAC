# from utils import patch_costcla
from rtac.utils.read_io import read_args
from rtac.utils.ensure_package import ensure_package
from rtac.rtac import rtac_factory
import sys
import os


# Absolute path to the current file
file_path = os.path.abspath(__file__)

# Directory containing the file
file_dir = os.path.dirname(file_path).split('examples')[0]


def main(scenario, instance_file):
    '''Run RAC process on, potentially infinite, problem instance sequence.'''

    instances = []
    with open(f'{instance_file}', 'r') as f:
        for line in f:
            instances.append(line.strip())

    rtac = rtac_factory(scenario)

    if scenario.gray_box:
        for i, instance in enumerate(instances):
            rtac.solve_instance(instance, next_instance=None)
            # If next problem instance arrives after rtac is started, it can be
            # passed while the configurator runs on current problem instance
            if i + 1 <= len(instances):
                rtac.provide_early_instance(instances[i + 1])
            # GB RAC needs to be wrapped up after running an iteration
            rtac.wrap_up_gb()
    else:
        for instance in instances:
            rtac.solve_instance(instance)


def run_example():
    ensure_package('python_tsp', '0.4.1')

    with open(f'{file_dir}/data/tsp_scenario_rt_test.txt', 'r') as f:
        lines = f.read().splitlines()

    new_lines = [
        f'--log_folder {file_dir}/logs',
        f'--feature_gen {file_dir}/feature_gen/tsp_feats.py',
        f'--param_file {file_dir}/data/params_tsp.json'
    ]

    lines += [line for line in new_lines if line not in lines]

    with open(f'{file_dir}/data/tsp_scenario_rt_test.txt', 'r') as f:
        lines = f.read().splitlines()

    scenario = read_args(f'{file_dir}/data/tsp_scenario_rt_test.txt', sys.argv)
    instance_file = f'{file_dir}/data/travellingsalesman_instances.txt'

    main(scenario, instance_file)


if __name__ == '__main__':
    pass
