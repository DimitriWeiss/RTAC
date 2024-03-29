from utils.read_io import read_args
from rtac import RTAC
import sys


def main():
    scenario = read_args('./data/tsp_scenario_q.txt', sys.argv)
    instance_file = './data/travellingsalesman_instances.txt'
    instances = []
    with open(f'{instance_file}', 'r') as f:
        for line in f:
            instances.append(line.strip())

    rtac = RTAC(scenario)

    for instance in instances:
        rtac.solve_instance(instance)


if __name__ == '__main__':
    main()
