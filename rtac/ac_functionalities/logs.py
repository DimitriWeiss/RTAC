"""This module contains functions for logging the data between
instances/tournaments as well as stats about toournaments and results."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import argparse
import os
import logging
import joblib
import numpy as np
import copy
from logging.handlers import RotatingFileHandler, BaseRotatingHandler
from rtac.ac_functionalities.rtac_data import (
    Configuration,
    RTACData,
    TournamentStats,
    ACMethod,
    Generator
)
__all__ = ('Configuration')


class NewRotatingFileHandler(RotatingFileHandler):
    """Overwriting logging.handlers.RotatingFileHandler in order to log to the
    same line in the file."""
    def __init__(self, filename, mode='w', maxBytes=0, backupCount=0):
        BaseRotatingHandler.__init__(self, filename, mode, encoding=None,
                                     delay=False)
        self.maxBytes = maxBytes
        self.backupCount = backupCount


class AbstractLogs(ABC):
    """Class with all functions and loggers concerning logging and loading
    RTAC and tournament data."""

    def __init__(self, scenario: argparse.Namespace):
        """Initializes logging class, check if log directory exists and create
        it if needed.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        """
        self.scenario = scenario
        self.ranking = scenario.ac
        if not os.path.isdir(scenario.log_folder):
            os.makedirs(scenario.log_folder)
        self.log_path = scenario.log_folder + '/' \
            + scenario.wrapper_name + '_' \
            + str(scenario.ac).split('.')[1]
        if scenario.gray_box:
            self.log_path += '_gb'
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        self.experimental = scenario.experimental
        if not scenario.resume:
            if not self.experimental:
                filelist = \
                    [f for f in os.listdir(self.log_path)
                     if f.endswith('.log')]
            else:
                filelist = \
                    [f for f in os.listdir(self.log_path)
                     if f.endswith('.log') and 'tourn_0' not in f]
            for f in filelist:
                os.remove(os.path.join(self.log_path, f))
        self.objective_min = scenario.objective_min
        print('\n')
        print(f'Logging to {self.log_path}')

    def init_rtac_logs(self) -> None:
        """Initializes loggers for realtime algorithm configuration data
        concerning all methods."""
        if not self.objective_min:
            self.times = {}
        else:
            self.results = {}
        # Set up general logging
        self.main_log = logging.getLogger('main_log')
        self.main_log.setLevel(logging.INFO)
        g_fh = logging.FileHandler(f'{self.log_path}/general.log')
        g_fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s',
                                      '%d/%m/%Y %H:%M:%S:')
        g_fh.setFormatter(formatter)
        self.main_log.addHandler(g_fh)

        # Set up winner trajectory logging
        self.winner_trajectory = logging.getLogger('winner_trajectory_log')
        self.winner_trajectory.setLevel(logging.INFO)
        wt_fh = logging.FileHandler(f'{self.log_path}/winner.log')
        wt_fh.setLevel(logging.INFO)
        wt_fh.setFormatter(formatter)
        self.winner_trajectory.addHandler(wt_fh)

        # Set up tournament stats logging
        self.tourn_stats_log = logging.getLogger('tourn_stats_log')
        self.tourn_stats_log.setLevel(logging.INFO)
        ts_fh = logging.FileHandler(f'{self.log_path}/tourn_stats.log')
        ts_fh.setLevel(logging.INFO)
        ts_fh.setFormatter(formatter)
        self.tourn_stats_log.addHandler(ts_fh)

        # Set up logging of the last tournament number
        self.tourn_nr_log = logging.getLogger('tourn_nr_log')
        self.tourn_nr_log.setLevel(logging.INFO)
        tn_fh = NewRotatingFileHandler(f'{self.log_path}/tourn_nr.log',
                                       mode='w', maxBytes=1, backupCount=0)
        tn_fh.setLevel(logging.INFO) 
        tn_fh.suffix = ""
        streamformatter = logging.Formatter(fmt='%(message)s')
        tn_fh.setFormatter(streamformatter)
        self.tourn_nr_log.addHandler(tn_fh)

        if not self.objective_min:
            # Set up ta winning runtime logging
            self.times_log = logging.getLogger('times_log')
            self.times_log.setLevel(logging.INFO)
            t_fh = NewRotatingFileHandler(f'{self.log_path}/times.log',
                                          mode='w', maxBytes=1, backupCount=0)
            t_fh.setLevel(logging.INFO) 
            t_fh.suffix = ""
            streamformatter = logging.Formatter(fmt='%(message)s')
            t_fh.setFormatter(streamformatter)
            self.times_log.addHandler(t_fh)

        else:
            # Set up ta results logging
            self.results_log = logging.getLogger('results_log')
            self.results_log.setLevel(logging.INFO)
            r_fh = NewRotatingFileHandler(f'{self.log_path}/results.log',
                                          mode='w', maxBytes=1, backupCount=0)
            r_fh.setLevel(logging.INFO) 
            r_fh.suffix = ""
            r_fh.setFormatter(streamformatter)
            self.results_log.addHandler(r_fh)

        # Set up tournament contender list logging
        self.contender_dict_log = logging.getLogger('contender_dict_log')
        self.contender_dict_log.setLevel(logging.INFO)
        cl_fh = \
            logging.FileHandler(f'{self.log_path}/contender_dict_tourn_0.log')
        cl_fh.setLevel(logging.INFO)
        self.contender_dict_log.addHandler(cl_fh)

    def general_log(self, message: str) -> None:
        """Log message.

        :param message: Any message provided.
        :type message: str
        """
        self.main_log.info(f'{message}')

    def scenario_log(self, scenario: argparse.Namespace) -> None:
        """Save scenario.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        """
        with open(f'{self.log_path}/scenario.log', 'w') as sf:
            sf.write(str(scenario))

    def rtac_log(self, rtac_data: RTACData,
                 tourn_stats: TournamentStats) -> None:
        """Logs for realtime algorithm configuration data
        concerning all methods.

        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        :param tourn_stats: Object containing statistics about the previous
            tournament.
        :type tourn_stats: TournamentStats
        """
        self.winner_trajectory.info(f'{rtac_data.winner.value}' + '\n')
        self.tourn_stats_log.info(str(tourn_stats) + '\n')
        self.tourn_nr_log.info(str(tourn_stats.tourn_nr) + '\n')
        if not self.objective_min:
            self.times[tourn_stats.id] = min(rtac_data.ta_res_time[:])
            self.times_log.info(str(self.times) + '\n')
        else:
            self.results[tourn_stats.id] = min(rtac_data.ta_res[:])
            self.results_log.info(str(self.results) + '\n')

    @abstractmethod
    def load_data(self) -> Any:
        """Loads the data of either last logged, or first tournament."""


class RTACLogs(AbstractLogs):
    """Class with all functions and loggers concerning logging and loading
    RTAC and tournament data for ReACTR implementation."""

    def init_ranking_logs(self) -> None:
        """Initializes loggers for data concerning ReACTR."""

        # Set up pool logging
        self.pool_log = logging.getLogger('pool_log')
        self.pool_log.setLevel(logging.INFO)
        p_fh = logging.FileHandler(f'{self.log_path}/pool_tourn_0.log')
        p_fh.setLevel(logging.INFO)
        self.pool_log.addHandler(p_fh)

        if self.ranking in (ACMethod.ReACTR, ACMethod.ReACTRpp):
            # Set up trueskill scores logging
            self.scores_log = logging.getLogger('scores_log')
            self.scores_log.setLevel(logging.INFO)
            s_fh = logging.FileHandler(f'{self.log_path}/scores_tourn_0.log')
            s_fh.setLevel(logging.INFO)
            self.scores_log.addHandler(s_fh)
        elif self.ranking is ACMethod.CPPL:
            # Set up bandit logging
            self.bandit_log = logging.getLogger('bandit_log')
            self.bandit_log.setLevel(logging.INFO)
            b_fh = logging.FileHandler(f'{self.log_path}/bandit_tourn_0.log')
            b_fh.setLevel(logging.INFO)
            self.bandit_log.addHandler(b_fh)

    def ranking_log(self, pool: dict[str: Configuration],
                    assessment: dict[str: Any], tourn_nr: int,
                    contender_dict: dict[str: Configuration],
                    **kwargs) -> None:
        """Logs data concerning ReACTR.

        :param pool: Dictionary with configuration id as key and configuration
            as value with scenario.contenders == #items .
        :type pool: dict
        :param assessment: Dictionary with configuration id as key and 
            assessment depending on the AC method used, e.g., trueskill scores,
            or bandit model.
        :type assessment: dict
        :param contender_dict: Dictionary with configuration id as key and
            configuration as value: contenders of the previous tournament.
        :type contender_dict: dict
        **kwargs : dict, optional
            Additional keyword arguments. Possible keys include:
            
            - `standard_scaler` sklearn
            - `min_max_scaler` sklearn
            - `pca_obj_params` sklearn
        """
        self.contender_dict_log.handlers.clear()
        cl_fh = logging.FileHandler(
            f'{self.log_path}/contender_dict_tourn_{tourn_nr}.log')
        cl_fh.setLevel(logging.INFO)
        self.contender_dict_log.addHandler(cl_fh)
        self.contender_dict_log.info(str(list(contender_dict.keys())))

        self.pool_log.handlers.clear()
        p_fh = logging.FileHandler(
            f'{self.log_path}/pool_tourn_{tourn_nr}.log')
        p_fh.setLevel(logging.INFO)
        self.pool_log.addHandler(p_fh)

        serializable_pool = copy.deepcopy(pool)
        for conf in serializable_pool.values():
            conf.gen = conf.gen.name
        self.pool_log.info(str(serializable_pool))

        if self.ranking in (ACMethod.ReACTR, ACMethod.ReACTRpp):
            self.scores_log.handlers.clear()
            s_fh = logging.FileHandler(
                f'{self.log_path}/scores_tourn_{tourn_nr}.log')
            s_fh.setLevel(logging.INFO)
            self.scores_log.addHandler(s_fh)
            self.scores_log.info(str(assessment))
        elif self.ranking is ACMethod.CPPL:
            if not os.path.isdir(f'{self.log_path}/bandit_models'):
                os.mkdir(f'{self.log_path}/bandit_models')
            self.bm_path = f'{self.log_path}/bandit_models'
            bandit_models = kwargs['bandit_models']
            if tourn_nr == 0:
                joblib.dump(
                    bandit_models['standard_scaler'],
                    f'{self.bm_path}/standard_scaler_{tourn_nr}.pkl')
                joblib.dump(
                    bandit_models['min_max_scaler'],
                    f'{self.bm_path}/min_max_scaler_{tourn_nr}.pkl')
                joblib.dump(
                    bandit_models['one_hot_encoder'],
                    f'{self.bm_path}/one_hot_encoder_{tourn_nr}.pkl')
                joblib.dump(
                    bandit_models['pca_obj_params'],
                    f'{self.bm_path}/pca_obj_params_{tourn_nr}.pkl')
                joblib.dump(
                    bandit_models['pca_obj_inst'],
                    f'{self.bm_path}/pca_obj_inst_{tourn_nr}.pkl')
            elif self.scenario.online_instance_train and tourn_nr > 0:
                joblib.dump(
                    bandit_models['standard_scaler'],
                    f'{self.bm_path}/standard_scaler_{tourn_nr}.pkl')
                joblib.dump(
                    bandit_models['pca_obj_inst'],
                    f'{self.bm_path}/pca_obj_inst_{tourn_nr}.pkl')
            elif len(bandit_models) == 0:
                pass
            self.bandit_log.handlers.clear()
            b_fh = logging.FileHandler(
                f'{self.log_path}/bandit_tourn_{tourn_nr}.log')
            b_fh.setLevel(logging.INFO)
            self.bandit_log.addHandler(b_fh)
            self.bandit_log.info(str(assessment))

    def parse_array(self, val):
        if not isinstance(val, str):
            return val

        val = val.strip()

        # Array-like: [1. 2. 3.]
        if val.startswith('[') and val.endswith(']'):
            val = val.strip("[]\n")
            return np.fromstring(val, sep=' ')

        # Scalar string: try to convert to int or float
        try:
            return int(val) if '.' not in val else float(val)
        except ValueError:
            raise ValueError(f"Cannot parse value: {val}")

    def load_data(self, tourn_nr: int | None = None) \
        -> tuple[dict[str: Configuration], dict[str: Any],
                 dict[str: Configuration], int,
                 Optional[Any]]:
        """Loads data necessary for resuming the algorithm configuration from
        last logged state of ReACTR.

        :returns: Configuration pool, scores and contender list of previously
            logged tournament, number of previously logged tournament.
        :rtype: tuple[dict[str: Configuration], dict[str: tuple[int, int]],
            dict[str: Configuration], int]
        """
        if tourn_nr is None:
            with open(f'{self.log_path}/tourn_nr.log') as f:
                tourn_nr = int(f.readline().strip())

        with open(f'{self.log_path}/pool_tourn_{tourn_nr}.log', 'r') as f:
            line = f.readline()
            pool = eval(line)
            for conf in pool.values():
                conf.gen = Generator[conf.gen]

        if self.ranking in (ACMethod.ReACTR, ACMethod.ReACTRpp):
            with open(
                    f'{self.log_path}/scores_tourn_{tourn_nr}.log', 'r') as f:
                assessment = eval(f.readline())
            if self.scenario.experimental:
                assessment = dict(zip(list(pool.keys()), assessment.values()))
        elif self.ranking is ACMethod.CPPL:
            self.bm_path = f'{self.log_path}/bandit_models'
            if not self.scenario.online_instance_train:
                tourn_nr = 0
            with open(
                    f'{self.log_path}/bandit_tourn_{tourn_nr}.log', 'r') as f:
                assessment = f.read()

            assessment = eval(assessment, {"array": np.array})
            assessment = \
                {k: self.parse_array(v) for k, v in assessment.items()}
            standard_scaler = \
                joblib.load(f'{self.bm_path}/standard_scaler_{tourn_nr}.pkl')
            min_max_scaler = \
                joblib.load(f'{self.bm_path}/min_max_scaler_0.pkl')
            one_hot_encoder = \
                joblib.load(f'{self.bm_path}/one_hot_encoder_0.pkl')
            pca_obj_params = \
                joblib.load(f'{self.bm_path}/pca_obj_params_0.pkl')
            pca_obj_inst = \
                joblib.load(f'{self.bm_path}/pca_obj_inst_{tourn_nr}.pkl')
            bandit_models = {'standard_scaler': standard_scaler,
                             'min_max_scaler': min_max_scaler,
                             'one_hot_encoder': one_hot_encoder,
                             'pca_obj_params': pca_obj_params,
                             'pca_obj_inst': pca_obj_inst}

        with open(f'{self.log_path}/contender_dict_tourn_{tourn_nr}.log',
                  'r') as f:
            contender_ids = eval(f.readline())

        if self.experimental:
            if self.ranking in (ACMethod.ReACTR, ACMethod.ReACTRpp):
                os.remove(f'{self.log_path}/scores_tourn_{tourn_nr}.log')
            elif self.ranking is ACMethod.CPPL:
                os.remove(f'{self.log_path}/bandit_tourn_{tourn_nr}.log')
                filelist = [f for f in os.listdir(self.bm_path)]
                for f in filelist:
                    os.remove(os.path.join(self.bm_path, f))
            os.remove(f'{self.log_path}/pool_tourn_{tourn_nr}.log')
            os.remove(f'{self.log_path}/contender_dict_tourn_{tourn_nr}.log')

        contender_dict = {}
        for ci in contender_ids:
            contender_dict[ci] = pool[ci]

        if self.ranking in (ACMethod.ReACTR, ACMethod.ReACTRpp):
            return pool, assessment, contender_dict, tourn_nr
        elif self.ranking is ACMethod.CPPL:
            return pool, assessment, contender_dict, tourn_nr, bandit_models


if __name__ == '__main__':
    pass
