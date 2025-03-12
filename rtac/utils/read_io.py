"""Loading RTAC scenario from file or sys.args into an argparse.Namespace."""

import argparse
import sys
import os
import warnings
import json
from ConfigSpace.read_and_write import pcs_new
from utils.json_validation import validateparams
from ac_functionalities.rtac_data import (
    ACMethod,
    ParamType,
    Parameter,
    ValType,
    Distribution,
    DiscreteParameter,
    ContinuousParameter,
    CategoricalParameter,
    BinaryParameter
)

sys.path.append(os.getcwd())


def translate_params(config_space: dict[str, dict]) \
        -> dict[str, DiscreteParameter | ContinuousParameter
                | CategoricalParameter | BinaryParameter]:
    """Translate configuration space nested dict to dict of dataclasses.

    :param config_space: Configuration space definition.
    :type config_space: dict of dicts

    :returns: config_space.
    :rtype: dict of dataclasses
    """
    for param, definition in config_space.items():
        config_space[param] = definition = argparse.Namespace(**definition)
        config_space[param] = \
            Parameter[definition.paramtype].value(**vars(definition))
        config_space[param].paramtype = ParamType[definition.paramtype]
        if config_space[param].paramtype \
                in (ParamType.categorical, ParamType.binary):
            config_space[param].valtype = ValType[config_space[param].valtype]
        else:
            if config_space[param].distribution is not Distribution.uniform:
                config_space[param].distribution = \
                    Distribution[definition.distribution]

    return config_space


def read_args(scenario: str = None,
              sysargs: list = None) -> argparse.Namespace:
    """Read in scenario arguments.

    :param scenario: Path to scenario text file.
    :type scenario: str
    :param sysargs: sys.argv passed from main.
    :type sysargs: list

    :returns: Scenario arguments set.
    :rtype: argparse.Namespace
    """
    if sysargs is not None:
        sysargs = list(sysargs[1:])
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--verbosity', type=int, 
                        default=0, 
                        help='Verbosity level. [0,1,2]')
    parser.add_argument('-n', '--number_cores', type=int, 
                        default=1,
                        help='Number of cores to be used in parallel.')
    parser.add_argument('-w', '--wrapper', type=str,
                        default='No wrapper chosen!', 
                        help='Module of Python wrapper for the algorithm.')
    parser.add_argument('-wn', '--wrapper_name', type=str,
                        default='No wrapper class name.',
                        help='''Name of the wrapper class in the wrapper
                        module.''')
    parser.add_argument('-fgn', '--feature_gen_name', type=str,
                        default='No feature generator class name.',
                        help='''Name of the feature generator class in the
                        feature generator mddule.''')
    parser.add_argument('-to', '--timeout', type=int, default=300, 
                        help='''Stop solving single instance after 
                        (int) seconds. [300]''')
    parser.add_argument('-c', '--contenders', type=int, default=30, 
                        help='The number of contenders in the pool. [30]')
    parser.add_argument('-kt', '--keeptop', type=int, default=2, 
                        help='''Number of top contenders to gbe part of
                        the tournament automatically for ReACTR/ReACTR++
                        (Rest is chosen randomly). [2]''')
    parser.add_argument('-ud', '--usedata', type=str, default=None, 
                        help='''Type y if data of prior run should 
                        be used. []''')
    parser.add_argument('-ch', '--chance', type=int, default=25, 
                        help='''Chance to replace gene randomly 
                        in percent (int: 0 - 100) for ReACTR/ReACTR++.
                        [25]''')
    parser.add_argument('-m', '--mutate', type=int, default=10, 
                        help='''Chance for mutation in crossover process 
                        in percent (int: 0 - 100) for ReACTR/ReACTR++. [10]''')
    parser.add_argument('-k', '--kill', type=float, default=5, 
                        help='''Contenders with a variance higher than 
                        this are killed and replaced (float) in
                        ReACTR/ReACTR++. [5]''')
    parser.add_argument('-tn', '--train_number', type=float, default=None, 
                        help='''How many of the first instances are to 
                        be trained on before starting (int). [None] ''')
    parser.add_argument('-tr', '--train_rounds', type=float, default=0, 
                        help='''How many rounds are the first -tn instances 
                        to be trained on (int). [1] ''')
    parser.add_argument('-fg', '--feature_gen', type=str,
                        default='', 
                        help='''Python wrapper to compute instance features
                        for given instance for CPPL/GBRAC.''')
    parser.add_argument('-npf', '--nc_pca_f', type=int, default=3, 
                        help='''Number of the dimensions for the PCA of the 
                        instance features for CPPL/GBRAC.''')
    parser.add_argument('-npp', '--nc_pca_p', type=int, default=5, 
                        help='''Number of the dimensions for the PCA of the 
                        parameter (features) for CPPL/GBRAC.''')
    parser.add_argument('-jfm', '--jfm', type=str, default='polynomial', 
                        help='''Mode of the joined feature map
                        for CPPL/GBRAC.''')
    parser.add_argument('-o', '--omega', type=float, default=0.0001, 
                        help='''Omega parameter for CPPL/GBRAC.''')
    parser.add_argument('-g', '--gamma', type=float, default=1, 
                        help='''Gamma parameter for CPPL/GBRAC.''')
    parser.add_argument('-a', '--alpha', type=float, default=0.2, 
                        help='''Alpha parameter for CPPL/GBRAC.''')
    parser.add_argument('-ac', '--ac', type=int, default='1', 
                        help='''Choice of Algorithm Configuration method.
                        Choose from: ReACTR, ReACTRpp, CPPL, Gray-Box by
                        [1, 2, 3, 4], respectively. ''')
    parser.add_argument('-pl', '--paramlimit', type=float, default=100000, 
                        help='''Limit for the possible absolute value of 
                        a parameter for it to be normed to log space 
                        before CPPL comptation.''')
    parser.add_argument('-ct', '--cpplt', type=int, default=8, 
                        help='''Reset value of t influencing Bandit 
                        Computations for CPPL/GBRAC.''')
    parser.add_argument('-lf', '--log_folder', type=str, 
                        default='logs', 
                        help='''Name of the directry to log in.''')
    parser.add_argument('-pf', '--param_file', type=str, 
                        default='No parameter file given!', 
                        help='''Path to the parameter file in PCS format or
                        RTAC json format.''')
    parser.add_argument('-r', '--resume',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='''Set \'-r\' or \'--resume\' flag to resume RTAC
                        from logged configuration state.''')
    parser.add_argument('-bp', '--baselineperf',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='''Set flag if only default 
                        parameterizations should run.''')
    parser.add_argument('-p', '--pws', action=argparse.BooleanOptionalAction,
                        default=False, 
                        help='''Inserting Default Configuration if set to
                        flag is set.''')
    parser.add_argument('-om', '--objective_min',
                        action=argparse.BooleanOptionalAction,
                        default=False, 
                        help='''Set flag \'-om\' or \'--objective_min\' if
                        optimizing for objective value minimization.''')
    parser.add_argument('-exp', '--experimental', 
                        action=argparse.BooleanOptionalAction,
                        default=False, 
                        help='''Set flag data of tournament 0 from logs are to
                        be used for experiment.''')

    # Read arguments from scenario file if provided and override them
    if scenario is not None:
        with open(f'{scenario}', 'r') as scenario_file:
            for line in scenario_file:
                sys.argv.extend(line.split())

    # Read arguments from sys.args if provided and override them
    if sysargs is not None:
        for i in range(0, len(sysargs), 2):
            if len(sysargs) > i + 1:  # Avoid typos in command line
                sys.argv.extend([sysargs[i], sysargs[i + 1]])

    scenario, unknown = parser.parse_known_args()

    if os.path.exists(f'{scenario.param_file}'):

        if '.json' in f'{scenario.param_file}':
        
            with open(f'{scenario.param_file}', 'r') as f:
                config_space = json.load(f)

            if validateparams(config_space):
                scenario.config_space = translate_params(config_space)
            else:
                warnings.warn('\nParameter definition is not valid!\n \
                    Add a valid json to scenario before starting \
                    configuration.')

        elif '.pcs' in f'{scenario.param_file}':
            with open(f'{scenario.param_file}', 'r') as f:
                scenario.config_space = pcs_new.read(f)

        else:
            warnings.warn(f'\nFile {scenario.param_file} does not exist!\n \
                Add a valid json or pcs to scenario before starting \
                configuration.')

    if len(unknown) > 0:
        us = str(set(unknown))[1:-1]
        warnings.warn(
            f'\n\nThe following arguments are unknown and ignored: {us}\n')

    scenario.ac = ACMethod(scenario.ac)

    return scenario


if __name__ == "__main__":
    pass
