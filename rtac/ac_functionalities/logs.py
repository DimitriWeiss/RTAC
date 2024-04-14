"""This module contains functions for logging the data between
instances/tournaments as well as stats about toournaments and results."""

from abc import ABC, abstractmethod
from typing import Any
import argparse
import os
import logging
from logging.handlers import RotatingFileHandler, BaseRotatingHandler
from ac_functionalities.rtac_data import (
    Configuration,
    RTACData,
    TournamentStats
)
__all__ = ('Configuration',)


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
        if not os.path.isdir(scenario.log_folder):
            os.makedirs(scenario.log_folder)
        self.log_path = scenario.log_folder + '/' \
            + scenario.wrapper_name + '_' \
            + str(scenario.ac).split('.')[1]
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

        # Set up trueskill scores logging
        self.scores_log = logging.getLogger('scores_log')
        self.scores_log.setLevel(logging.INFO)
        s_fh = logging.FileHandler(f'{self.log_path}/scores_tourn_0.log')
        s_fh.setLevel(logging.INFO)
        self.scores_log.addHandler(s_fh)

    def ranking_log(self, pool: dict[str: Configuration],
                    scores: dict[str: tuple[int, int]], tourn_nr: int,
                    contender_dict: dict[str: Configuration]) -> None:
        """Logs data concerning ReACTR.

        :param pool: Dictionary with configuration id as key and configuration
            as value with scenario.contenders == #items .
        :type pool: dict
        :param scores: Dictionary with configuration id as key and tuple of Mu
            and Sigma as trueskill performance assessments as value.
        :type scores: dict
        :param contender_dict: Dictionary with configuration id as key and
            configuration as value: contenders of the previous tournament.
        :type contender_dict: dict
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
        self.pool_log.info(str(pool))

        self.scores_log.handlers.clear()
        s_fh = logging.FileHandler(
            f'{self.log_path}/scores_tourn_{tourn_nr}.log')
        s_fh.setLevel(logging.INFO)
        self.scores_log.addHandler(s_fh)
        self.scores_log.info(str(scores))

    def load_data(self, tourn_nr: int | None = None) \
        -> tuple[dict[str: Configuration], dict[str: tuple[int, int]],
                 dict[str: Configuration], int]:
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
            pool = eval(f.readline())

        with open(f'{self.log_path}/scores_tourn_{tourn_nr}.log', 'r') as f:
            scores = eval(f.readline())

        with open(f'{self.log_path}/contender_dict_tourn_{tourn_nr}.log',
                  'r') as f:
            contender_ids = eval(f.readline())

        if self.experimental:
            os.remove(f'{self.log_path}/pool_tourn_{tourn_nr}.log')
            os.remove(f'{self.log_path}/scores_tourn_{tourn_nr}.log')
            os.remove(f'{self.log_path}/contender_dict_tourn_{tourn_nr}.log')

        contender_dict = {}
        for ci in contender_ids:
            contender_dict[ci] = pool[ci]

        return pool, scores, contender_dict, tourn_nr
