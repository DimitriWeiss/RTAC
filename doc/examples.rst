.. _examples:

Examples
========

To run ReACTR, you can use the following scenario file:

.. code-block:: text

    --verbosity 2
    --number_cores 8
    --timeout 120
    --contenders 30
    --keeptop 2
    --pws
    --experimental
    --chance 25
    --mutate 10
    --kill 5
    --ac 1
    --paramlimit 100000
    --log_folder logs
    --wrapper wrapper.cadical
    --wrapper_name Cadicalpp
    --feature_gen absoulte_path_to/feature_gen/cadical_feats.py
    --param_file absoulte_path_to/data/params_cadical.json


You can then use a python script in the following manner to solve incoming problem instances:

.. code-block:: python

    from rtac.utils.read_io import read_args
    from rtac.rtac import rtac_factory
    import sys


    def main(scenario, instance_file):
        '''Run RAC process on, potentially infinite, problem instance sequence.'''

        instances = []
        with open(f'{instance_file}', 'r') as f:
            for line in f:
                instances.append(line.strip())

        rtac = rtac_factory(scenario)

        for instance in instances:
            rtac.solve_instance(instance)


    if __name__ == '__main__':
        scenario = read_args('./scenario.txt', sys.argv)
        instance_file = './instance_sequence.txt'

        main(scenario, instance_file)


Of course, you can adjust the script to wait and receive problem instances to be passed to rtac.solve_instance.