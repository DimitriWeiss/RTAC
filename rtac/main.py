from utils.read_io import read_args
from rtac import rtac_factory
import sys


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
            rtac.provide_early_instance(instances[i + 1])
            # GB RAC needs to be wrapped up after running an iteration
            rtac.wrap_up_gb()
    else:
        for instance in instances:
            rtac.solve_instance(instance)


if __name__ == '__main__':
    # scenario = read_args('./data/tsp_scenario_rt_cppl.txt', sys.argv)
    # scenario = read_args('./data/tsp_scenario_rt_cppl_gb.txt', sys.argv)
    # scenario = read_args('./data/tsp_scenario_rt.txt', sys.argv)
    # scenario = read_args('./data/tsp_scenario_rt_pp.txt', sys.argv)
    # scenario = read_args('./data/cadical_scenario.txt', sys.argv)
    # scenario = read_args('./data/cadical_scenario_rt_cppl.txt', sys.argv)
    # scenario = read_args('./data/cadical_scenario_rt_cppl_100.txt', sys.argv)
    scenario = read_args('./data/cadical_scenario_rt_cppl_100_gb.txt', sys.argv)
    # instance_file = './data/travellingsalesman_instances.txt'
    # instance_file = './data/power_law_easy_instances.txt'
    instance_file = './data/power_law_SAT_drift_100.txt'
    # instance_file = './data/power_law_SAT_drift_1000.txt'

    main(scenario, instance_file)
