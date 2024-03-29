import python_tsp.heuristics.simulated_annealing as tsp
import argparse
import json
import scipy
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", "--instance",
                        type=str,
                        help='Problem instance.')
    parser.add_argument("-ps", "--perturbation_scheme",
                        type=str,
                        default='two_opt',
                        help="Perturbation scheme.")
    parser.add_argument("-a", "--alpha",
                        type=float,
                        default=0.9,
                        help='Annealing rate [0.0001, 0.9999].')
    parser.add_argument("-t", "--timeout",
                        type=float,
                        default=300,
                        help='Time limit.')
    parser.add_argument("-mni", "--MAX_NON_IMPROVEMENTS",
                        type=int,
                        default=3,
                        help='Maximum steps without improvement [1, 100].')
    parser.add_argument("-miim", "--MAX_INNER_ITERATIONS_MULTIPLIER",
                        type=int,
                        default=10,
                        help='Maximum inner iterations multiplier [1, 100].')

    args = parser.parse_args()

    with open(f'{args.instance}') as f:
        coordinates = json.load(f)

    nodes = []

    for i, _ in enumerate(coordinates['xs']):
        nodes.append([coordinates['xs'][i], coordinates['ys'][i]])

    distance_matrix = scipy.spatial.distance_matrix(nodes, nodes)

    start = time.perf_counter()

    tsp.MAX_NON_IMPROVEMENTS = args.MAX_NON_IMPROVEMENTS
    tsp.MAX_INNER_ITERATIONS_MULTIPLIER = args.MAX_INNER_ITERATIONS_MULTIPLIER

    permutation, distance = \
        tsp.solve_tsp_simulated_annealing(
            distance_matrix, perturbation_scheme=args.perturbation_scheme,
            alpha=args.alpha, max_processing_time=args.timeout, verbose=True)

    print('Time:', time.perf_counter() - start)
    print('Distance:', distance)
