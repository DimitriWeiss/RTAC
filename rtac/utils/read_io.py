"""Loading RTAC scenario from file or sys.args into an argparse.Namespace."""

from utils.background_thread_control import set_background_thread_nr

set_background_thread_nr()

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
    parser.add_argument('-fdp', '--feature_path', type=str,
                        default='No feature directory path.',
                        help='''Path to the directory with feature files,
                        if existing.''')
    parser.add_argument('-to', '--timeout', type=int, default=300, 
                        help='''Stop solving single instance after 
                        (int) seconds. [300]''')
    parser.add_argument('-rp', '--runtimePAR', type=int, default=1, 
                        help='''Multiply -to by -rp,
                        if instance not solved. [1]''')
    parser.add_argument('-c', '--contenders', type=int, default=30, 
                        help='The number of contenders in the pool. [30]')
    parser.add_argument('-kt', '--keeptop', type=int, default=4, 
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
    parser.add_argument('-fg', '--feature_gen', type=str,
                        default='', 
                        help='''Python wrapper to compute instance features
                        for given instance for CPPL/GBRAC.''')
    parser.add_argument('-ipt', '--instance_pre_train', type=str,
                        default=False, 
                        help='''If the bandit ought to be pre-trained with 
                        a set of instance features, provide the path to a
                        test file with lists of features in each line, after
                        this argument.''')
    parser.add_argument('-npf', '--nc_pca_f', type=int, default=3, 
                        help='''Number of the dimensions for the PCA of the 
                        instance features for CPPL/GBRAC.''')
    parser.add_argument('-npp', '--nc_pca_p', type=int, default=5, 
                        help='''Number of the dimensions for the PCA of the 
                        parameter (features) for CPPL/GBRAC.''')
    parser.add_argument('-jfm', '--jfm', type=str, default='polynomial', 
                        help='''Mode of the joined feature map
                        for CPPL/GBRAC.''')
    parser.add_argument('-o', '--omega', type=float, default=1.0, 
                        help='''Omega parameter for CPPL/GBRAC.''')
    parser.add_argument('-g', '--gamma', type=float, default=1, 
                        help='''Gamma parameter for CPPL/GBRAC.''')
    parser.add_argument('-a', '--alpha', type=float, default=0.2, 
                        help='''Alpha parameter for CPPL/GBRAC.''')
    parser.add_argument('-e', '--epsilon', type=float, default=0.9, 
                        help='''Epsilopn for epsilon-greedy selection. Set to
                         0.0 for greedy selection. Setting epsilon avoids 
                         convergence of confidence through oversampling 
                         of a set of arms by forcing the sampling of rarely 
                         seen arms.''')
    parser.add_argument('-ka', '--kappa', type=float, default=1.0, 
                        help='''Weight on confidence in pairwise comparison
                        for discarding configurations from pool in CPPL.''')
    parser.add_argument('-gen_mult', '--gen_mult', type=int, default=2, 
                        help='''Factor by which to multiply the number of 
                        configurations being generated before assessed by CPPL 
                        to be inserted into pool.''')
    parser.add_argument('-ac', '--ac', type=int, default='1', 
                        help='''Choice of Algorithm Configuration method.
                        Choose from: ReACTR, ReACTRpp, CPPL, Gray-Box by
                        [1, 2, 3, 4], respectively. ''')
    parser.add_argument('-pl', '--paramlimit', type=float, default=100000, 
                        help='''Limit for the possible absolute value of 
                        a parameter for it to be normed to log space 
                        before CPPL comptation.''')
    parser.add_argument('-wb', '--win_bonus', type=float, default=0.2, 
                        help='''Factor by which recent CPPL winner skill
                        is boosted.''')
    parser.add_argument('-wd', '--win_decay', type=float, default=0.8, 
                        help='''Factor by which CPPL skill estimates are
                        decaying over the course of the tournaments.''')
    parser.add_argument('-rwb', '--recent_winner_boost', type=float,
                        default=3.0, 
                        help='''Absolute number by which the skill estimate
                        of a recent CPPL winner is boosted.''')
    parser.add_argument('-ff', '--forgetting_factor', type=float,
                        default=0.98, 
                        help='''Factor by which old observations are forgotten
                        (theta_bar in CPPL). Necessary, since pool changes.''')
    parser.add_argument('-on', '--obs_noise', type=float,
                        default=1.0, 
                        help='''Assumed obervation noise variance for theta_bar
                         in CPPL.''')
    parser.add_argument('-cr', '--cppl_reward', type=float,
                        default=1.0, 
                        help='''Reward of a winning arm in CPPL bandit model.
                        ''')
    parser.add_argument('-cpplt', '--cpplt', type=float,
                        default=2, 
                        help='''Value to which t in the CPPL bandit is reset to
                         if the pool is changed.''')
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
    parser.add_argument('-oit', '--online_instance_train', 
                        action=argparse.BooleanOptionalAction,
                        default=False, 
                        help='''Enable condinued PCA fitting on incoming
                        problem instances.''')
    parser.add_argument('-epsg', '--epsilon_greedy', 
                        action=argparse.BooleanOptionalAction,
                        default=False, 
                        help='''Enable epsilon greedy selection.''')
    parser.add_argument('-ib', '--isolate_bandit', 
                        action=argparse.BooleanOptionalAction,
                        default=False, 
                        help='''Runs bandit class in child processes to free 
                        resources after use. Useful if if bandit resource use 
                        collides with algorithm resource use but incurs higher 
                        overhead due to pickling and loading of the bandit 
                        class.''')
    parser.add_argument('-gb', '--gray_box', 
                        action=argparse.BooleanOptionalAction,
                        default=False, 
                        help='''Enable gray-box RAC.''')
    parser.add_argument('-gbrt', '--gb_read_time', type=float, default=0.1, 
                        help='''Freuqency in which to check for gray-box 
                        output in seconds.''')
    parser.add_argument('-ngbf', '--nr_gb_feats', type=int, default=2, 
                        help='''Number of gray-box features used.''')

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
