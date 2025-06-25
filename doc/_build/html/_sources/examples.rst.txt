.. _examples:

Examples
========

Example Target Algorithm Wrapper
--------------------------------

With this example, the target algorithm to be configured needs to be callable via command line and accept parameters via command line. It also has to provide runtime output to the terminal in order to be used for the gray-box extension.

.. code-block:: python

    from subprocess import Popen, PIPE
    import subprocess
    from typing import Any
    import time
    import sys
    import os
    from rtac.wrapper.abstract_wrapper import AbstractWrapper
    from rtac.ac_functionalities.rtac_data import Configuration, InterimMeaning
    from rtac.ac_functionalities.ta_runner import non_block_read

    sys.path.append(os.getcwd())


    class TSP_RT(AbstractWrapper):
        """TSP wrapper for runtime scenario."""

        def translate_config(self, config: Configuration) -> list[str]:

            config_list = []
            config.conf['-a'] = 0.9  # runtime scenario: fixed annealing factor
            for name, param in config.conf.items():
                config_list.append(name)
                config_list.append(str(param))

            return config_list

        def start(self, config: Any, timeout: int,
                  instance: str) -> tuple[subprocess.Popen, int]:

            # Absolute path to the current file
            file_path = os.path.abspath(__file__)

            # Directory containing the file
            file_dir = os.path.dirname(file_path)
            file_dir = file_dir.split('wrapper')[0]

            proc = Popen(['python3',
                          f'{file_dir}data/solvers/python-tsp.py',
                          *config, '-t', str(timeout), '-i',
                          f'{file_dir}{instance}'],
                         stdout=PIPE)

            self.timeout = timeout

            proc_cpu_time = time.process_time_ns()

            return proc, proc_cpu_time

        def check_if_solved(self, ta_output: bytes, nnr: non_block_read,
                            proc: subprocess.Popen) -> tuple[
                                int | float, float, int] | None:

            if ta_output != b'':
                b = str(ta_output.strip())
                if 'Warning' in b:  # Appears in b, if TA reaches time limit
                    time = self.timeout
                    res = sys.maxsize
                    event = 0

                    return res, time, event

                if 'Time:' in b:
                    time = float(b.split(' ')[1][:-1])
                    res_not_given = True
                    while res_not_given:
                        line = nnr(proc.stdout)
                        b = str(line.strip())
                        if 'Distance:' in b:
                            res = float(b.split(' ')[1][:-1])
                            res_not_given = False

                    event = 1
                    proc.stdout.close()

                else:

                    return None

                return res, time, event
            else:
                return None


    class TSP_Q(TSP_RT):
        """TSP wrapper for solution quality scenario."""

        def translate_config(self, config: Configuration) -> list[str]:

            config_list = []
            for name, param in config.conf.items():
                config_list.append(name)
                config_list.append(str(param))

            return config_list


    class TSP_RTpp(TSP_RT):
        """TSP wrapper for runtime scenario with runtime output."""

        def interim_info(self) -> list[InterimMeaning]:

            self.interim_meaning = [InterimMeaning.decrease]

            return self.interim_meaning

        def check_output(self, ta_output: bytes) -> list[float] | None:

            if ta_output != b'':
                b = str(ta_output.strip())
                # Check for progress
                if 'Temperature' in b:
                    b = b.split(' ')
                    # Assumption: the lower the temperature, the closer the TA is
                    # to finding the solution. Solution Quality is not regarded in
                    # this example, we optimize for runtime.
                    temp = float(b[1][:-1])
                    interim = [temp]

                    return interim
                else:
                    return None
            else:
                return None


    class TSP_Qpp(TSP_Q):
        """TSP wrapper for solution quality scenario with runtime output."""

        def interim_info(self) -> list[InterimMeaning]:

            self.interim_meaning = [InterimMeaning.decrease,
                                    InterimMeaning.increase,
                                    InterimMeaning.decrease,
                                    InterimMeaning.increase]

            return self.interim_meaning

        def check_output(self, ta_output) -> list[float] | None:

            if ta_output != b'':
                b = str(ta_output.strip())
                # Check for progress
                if 'Temperature' in b:
                    b = b.split(' ')
                    
                    temp = float(b[1][:-1])
                    k = float(b[6].split('/')[0])
                    k_acc = float(b[8].split('/')[0])
                    k_noimp = float(b[10][:-1])
                    interim = [temp, k, k_acc, k_noimp]

                    return interim
                else:
                    return None
            else:
                return None

Scenario File Examples
----------------------

ReACTR
~~~~~~

To run ReACTR, you can use a scenario file like this:

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
    --wrapper wrapper.tsp
    --wrapper_name TSP
    --feature_gen absoulte_path_to/feature_gen/tsp_feats.py
    --param_file absoulte_path_to/data/tsp_params.json


ReACTR++
~~~~~~~~

To run ReACTR++, you can use a scenario file like this:

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
    --ac 2
    --paramlimit 100000
    --log_folder logs
    --wrapper wrapper.tsp
    --wrapper_name TSPpp
    --feature_gen absoulte_path_to/feature_gen/tsp_feats.py
    --param_file absoulte_path_to/data/tsp_params.json


CPPL
~~~~

To run CPPL, you can use a scenario file like this:

.. code-block:: text

    --verbosity 2
    --number_cores 8
    --timeout 120
    --runtimePAR 1
    --contenders 30
    --keeptop 4
    --pws
    --experimental
    --chance 25
    --mutate 10
    --online_instance_train
    --nc_pca_f 3
    --nc_pca_p 5
    --jfm polynomial
    --omega 1.0
    --gamma 1
    --alpha 0.2
    --epsilon 0.9
    --kappa 1.0
    --ac 3
    --paramlimit 100000
    --epsilon_greedy
    --gen_mult 2
    --log_folder logs
    --wrapper wrapper.tsp
    --wrapper_name TSPpp
    --feature_gen feature_gen.tsp_feats
    --feature_gen_name TSPFeats
    --feature_path absoulte_path_to/feature_gen/cad_features/Features_tsp.csv
    --param_file absoulte_path_to/data/tsp_params.json
    --instance_pre_train absoulte_path_to/feature_gen/tsp_features/pre_train_features.txt

You do not need to provide --feature_path if you are computing problem instance features online. Pre-train features need to be in a text file as Python lists in an individual line per instance.


Gray-Box
~~~~~~~~

To run CPPL in gray-box mode, you can use a scenario file like this:

.. code-block:: text

    --verbosity 2
    --number_cores 8
    --timeout 120
    --runtimePAR 1
    --contenders 30
    --keeptop 4
    --pws
    --experimental
    --chance 25
    --mutate 10
    --online_instance_train
    --nc_pca_f 3
    --nc_pca_p 5
    --jfm polynomial
    --omega 1.0
    --gamma 1
    --alpha 0.2
    --epsilon 0.9
    --kappa 1.0
    --ac 3
    --paramlimit 100000
    --epsilon_greedy
    --gen_mult 2
    --log_folder logs
    --gray_box
    --gb_read_time 0.1
    --nr_gb_feats 2
    --wrapper wrapper.tsp
    --wrapper_name TSPpp
    --feature_gen feature_gen.tsp_feats
    --feature_gen_name TSPFeats
    --feature_path absoulte_path_to/feature_gen/cad_features/Features_tsp.csv
    --param_file absoulte_path_to/data/tsp_params.json
    --instance_pre_train absoulte_path_to/feature_gen/tsp_features/pre_train_features.txt

You do not need to provide --feature_path if you are computing problem instance features online. You can run ReACTR and ReACTR++ in gray-box mode by adding --gray_box and --nr_gb_feats and using a wrapper with runtime output mechanism. Pre-train features need to be in a text file as Python lists in an individual line per instance.

Parameter Files
---------------

PCS
~~~

You can find documentation for PCS at `PCS Documentation <https://automl.github.io/ConfigSpace/latest/>`_. Here is an example:

.. code-block:: text

    ps categorical {ps1, ps2, ps3, ps4, ps5, ps6, two_opt} [two_opt]
    a real [0.0001, 0.9999] [0.9]
    mni integer [1, 100] [3]
    miim integer [1, 100] [10]


JSON
~~~~

RTAC has an own json schema for parameter space definition.


Each parameter is defined as a dictionary key in a JSON object.

Required Field
^^^^^^^^^^^^^^

- **``paramtype``**: must be one of ``"categorical"``, ``"discrete"``, ``"continuous"``, or ``"binary"``.

Fields by Type
^^^^^^^^^^^^^^

- **Continuous**
  
  - Requires: ``minval``, ``maxval`` (both numbers), ``default`` (number)

- **Discrete**
  
  - Requires: ``minval``, ``maxval`` (both integers), ``default`` (integer)

- **Binary**
  
  - Requires: ``default`` (integer or string)

- **Categorical**
  
  - If ``valtype`` is ``"int"``:
  
    - Requires: ``minval``, ``maxval`` (integers), ``valtype``
  
  - If ``valtype`` is ``"str"``:
  
    - Requires: ``values`` (array of strings), ``valtype``
    - Optional: ``default`` (string)

Optional Common Fields
^^^^^^^^^^^^^^^^^^^^^^

- ``default``: default value (number, integer, or string depending on type)
- ``flag``: if ``true``, this is a binary flag, not a parameter
- ``distribution``: either ``"uniform"`` or ``"log"`` (defaults to uniform)

Log Distribution Options (if used)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Requires ``distribution = "log"``, and optionally allows:

- ``splitbydefault``: split log range at default value
- ``logonpos`` / ``logonneg``: enable log sampling on positive/negative side (if ``minval < 0``)
- ``probabpos``, ``probabneg``: probability of sampling positive/negative side (between 0 and 1, exclusive)
- ``includezero``: include zero in log sampling
- ``probabilityzero``: probability of choosing zero (between 0 and 1, exclusive)

Example
^^^^^^^

{'param_6mdb': {'paramtype': 'discrete',
  'minval': 0,
  'maxval': 10,
  'default': 0},
 'param_98dt': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_g3al': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_4nie': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_f3ed': {'paramtype': 'categorical',
  'valtype': 'int',
  'minval': 1,
  'maxval': 3,
  'default': 3},
 'param_bcnn': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_3zry': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'false'},
 'param_7oqx': {'paramtype': 'discrete',
  'minval': 0,
  'maxval': 2000000000,
  'default': 100000,
  'distribution': 'log',
  'includezero': True,
  'probabilityzero': 0.05,
  'logonpos': True,
  'probabpos': 0.9},
 'param_gids': {'paramtype': 'discrete',
  'minval': 2,
  'maxval': 2000000000,
  'default': 4,
  'distribution': 'log',
  'logonpos': True,
  'probabpos': 0.9},
 'param_1dly': {'paramtype': 'discrete',
  'minval': 0,
  'maxval': 2000000000,
  'default': 100,
  'distribution': 'log',
  'includezero': True,
  'probabilityzero': 0.05,
  'logonpos': True,
  'probabpos': 0.9},
 'param_t43b': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_jnjw': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_mb9s': {'paramtype': 'discrete',
  'minval': 1,
  'maxval': 3,
  'default': 1},
 'param_6o9q': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'false'},
 'param_ovbg': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_luvg': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_p9dc': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'false'},
 'param_g0aj': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_hlcw': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_imgg': {'paramtype': 'discrete',
  'minval': 0,
  'maxval': 2,
  'default': 1},
 'param_hpvo': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'false'},
 'param_bc51': {'paramtype': 'discrete',
  'minval': 0,
  'maxval': 2000000000,
  'default': 100,
  'distribution': 'log',
  'splitbydefault': True,
  'includezero': True,
  'probabilityzero': 0.05,
  'logonpos': True,
  'probabpos': 0.9},
 'param_6040': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_m6pf': {'paramtype': 'categorical',
  'valtype': 'str',
  'values': ['true', 'false'],
  'default': 'true'},
 'param_wu3s': {'paramtype': 'discrete',
  'minval': 1,
  'maxval': 2000000000,
  'default': 2000,
  'distribution': 'log',
  'splitbydefault': True,
  'logonpos': True,
  'probabpos': 0.8}}

Example RTAC call
-----------------

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


    if __name__ == '__main__':
        scenario = read_args('./scenario.txt', sys.argv)
        instance_file = './instance_sequence.txt'

        main(scenario, instance_file)



Of course, you can adjust the script to wait and receive problem instances to be passed to rtac.solve_instance.