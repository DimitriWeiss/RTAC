"""This module contans classes for touenament management according to the RTAC
method used."""

from abc import ABC, abstractmethod
import argparse
import time
from ac_functionalities.tournament import Tournament
from ac_functionalities.result_processing import processing_factory
from ac_functionalities.rtac_data import TARunStatus, RTACData, ACMethod
from ac_functionalities.ta_runner import BaseTARunner
from ac_functionalities.logs import RTACLogs


class AbstractTournamentManager(ABC):
    """Abstract tournament manager class."""

    def __init__(self, scenario: argparse.Namespace, ta_runner: BaseTARunner,
                 logs: RTACLogs, rtac_data: RTACData) -> None:
        """Initialize tournamnt management class with data and objects
        necessary for RTAC method used. If self.scenario.resume, data of last
        logged tournament is loaded and algorithm configuration is performed
        from that state.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        :param ta_runner: Target algorithm runner object.
        :type: BaseTARunner
        :param logs: Object containing loggers and logging functions.
        :type: RTACLogs
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        """
        self.scenario = scenario
        self.ta_runner = ta_runner
        self.rtac_data = rtac_data
        self.logs = logs

        self.tournament = Tournament(self.scenario, self.ta_runner,
                                     self.rtac_data, self.logs)

        self.res_process = processing_factory(self.scenario, self.logs)

        if self.scenario.resume:
            if self.scenario.ac in (ACMethod.ReACTR, ACMethod.ReACTRpp):
                self.res_process.pool, self.res_process.scores, \
                    self.contender_dict, self.tourn_nr = self.logs.load_data()
            elif self.scenario.ac is ACMethod.CPPL:
                self.res_process.pool, self.res_process.scores, \
                    self.contender_dict, self.tourn_nr, \
                    self.bandit_models = self.logs.load_data()
            self.res_process.contender_dict = self.contender_dict
            self.scenario.resume = False

        elif self.scenario.experimental:
            print('self.scenario.experimental', self.scenario.experimental)
            if self.scenario.ac in (ACMethod.ReACTR, ACMethod.ReACTRpp):
                self.res_process.pool, self.res_process.scores, \
                    self.contender_dict, self.tourn_nr \
                    = self.logs.load_data(tourn_nr=0)
                self.res_process.contender_dict = self.contender_dict
            elif self.scenario.ac is ACMethod.CPPL:
                self.res_process.pool, self.res_process.bandit, \
                    self.contender_dict, self.tourn_nr, \
                    self.bandit_models = self.logs.load_data(tourn_nr=0)
                self.res_process.contender_dict = self.contender_dict
                #self.res_process.contender_dict = \
                #    self.res_process.cppl.contender_dict = self.contender_dict
                #self.res_process.cppl.pool = self.res_process.pool

        else:
            self.tourn_nr = 0
            self.contender_dict = self.res_process.get_contender_dict()

        self.logs.init_rtac_logs()
        self.logs.init_ranking_logs()

    @abstractmethod
    def set_tourn_status(self) -> None:
        """Setting the results of the tournament and status of the target
        algorithm runs to the rtac_data.TournamentStats object according to
        the RTAC method used."""

    @abstractmethod
    def solve_instance(self, instance: str, rtac_data: RTACData) -> RTACData:
        """Solving the problem instance according to the RTAC method used.
        :param instance: path to the problem instance to solve.
        
        :type instance: str
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        :returns: Updated object containing data and objects necessary
            throughout the rtac modules
        :rtype: RTACData
        """


class TournamentManager(AbstractTournamentManager):
    """Tournament manager class for the ReACTR implementation."""

    def set_tourn_status(self) -> None:
        """Setting the results of the tournament and status of the target
        algorithm runs to the rtac_data.TournamentStats object."""
        self.tourn_stats.results = self.rtac_data.ta_res[:]
        self.tourn_stats.times = self.rtac_data.ta_res_time[:]
        self.tourn_stats.rtac_times = self.rtac_data.ta_rtac_time[:]
        for tr, tarun in enumerate(self.tourn_stats.TARuns):
            self.tourn_stats.TARuns[tarun].res = self.rtac_data.ta_res[tr]
            self.tourn_stats.TARuns[tarun].time = \
                self.rtac_data.ta_res_time[tr]
            self.tourn_stats.TARuns[tarun].status = \
                TARunStatus(self.rtac_data.status[tr])

    def solve_instance(self, instance: str, rtac_data: RTACData) -> RTACData:
        """Solving the problem instance according to the ReACTR implementation.
        :param instance: path to the problem instance to solve.

        :type instance: str
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        :returns: Updated object containing data and objects necessary
            throughout the rtac modules
        :rtype: RTACData
        """
        self.instance = instance
        self.rtac_data = self.tournament.rtac_data = rtac_data
        self.logs.ranking_log(self.res_process.pool, self.res_process.scores,
                              self.tourn_nr, self.contender_dict)

        self.tournament.start_tournament(self.instance, self.contender_dict,
                                         self.tourn_nr)
        self.tournament.watch_tournament()

        # Update tournament status
        self.tourn_stats = self.tournament.tournamentstats
        self.tourn_stats.winner = self.rtac_data.winner.value
        self.res_process.process_tourn(self.rtac_data)
        if self.scenario.verbosity == 2:
            message = 'Results:' + str(self.rtac_data.ta_res[:]) + '\n' \
                + 'Runtimes: ' + str(self.rtac_data.ta_res_time[:]) + '\n' \
                + 'Runtimes with overhead: ' \
                + str(self.rtac_data.ta_rtac_time[:])
            print(message)
            self.logs.general_log(message)
        self.set_tourn_status()
        self.logs.rtac_log(self.rtac_data, self.tourn_stats)
        log_message = f'Winner of tournament {self.tournament.tourn_id}' \
                      + f' (nr. {self.tourn_nr}) is {self.tourn_stats.winner}'
        self.logs.general_log(log_message)
        
        self.contender_dict = self.res_process.get_contender_dict()

        self.tourn_nr += 1

        return self.rtac_data


class TournamentManagerCPPL(AbstractTournamentManager):
    """Tournament manager class for the CPPL implementation."""

    def set_tourn_status(self) -> None:
        """Setting the results of the tournament and status of the target
        algorithm runs to the rtac_data.TournamentStats object."""
        self.tourn_stats.results = self.rtac_data.ta_res[:]
        self.tourn_stats.times = self.rtac_data.ta_res_time[:]
        self.tourn_stats.rtac_times = self.rtac_data.ta_rtac_time[:]
        for tr, tarun in enumerate(self.tourn_stats.TARuns):
            self.tourn_stats.TARuns[tarun].res = self.rtac_data.ta_res[tr]
            self.tourn_stats.TARuns[tarun].time = \
                self.rtac_data.ta_res_time[tr]
            self.tourn_stats.TARuns[tarun].status = \
                TARunStatus(self.rtac_data.status[tr])

    def select_contenders(self):
        self.res_process.select_contenders(self.instance)

    def solve_instance(self, instance: str, rtac_data: RTACData) -> RTACData:
        """Solving the problem instance according to the ReACTR implementation.
        :param instance: path to the problem instance to solve.

        :type instance: str
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        :returns: Updated object containing data and objects necessary
            throughout the rtac modules
        :rtype: RTACData
        """
        self.instance = instance
        self.rtac_data = self.tournament.rtac_data = rtac_data

        self.logs.ranking_log(self.res_process.pool, self.res_process.bandit,
                              self.tourn_nr, self.contender_dict,
                              bandit_models=self.res_process.bandit_models)

        self.tournament.start_tournament(self.instance, self.contender_dict,
                                         self.tourn_nr)
        self.tournament.watch_tournament()

        # Update tournament status
        self.tourn_stats = self.tournament.tournamentstats
        self.tourn_stats.winner = self.rtac_data.winner.value
        self.res_process.process_tourn(self.rtac_data, self.instance)
        if self.scenario.verbosity == 2:
            message = 'Results:' + str(self.rtac_data.ta_res[:]) + '\n' \
                + 'Runtimes: ' + str(self.rtac_data.ta_res_time[:]) + '\n' \
                + 'Runtimes with overhead: ' \
                + str(self.rtac_data.ta_rtac_time[:])
            print(message)
            self.logs.general_log(message)
        self.set_tourn_status()
        self.logs.rtac_log(self.rtac_data, self.tourn_stats)
        log_message = f'Winner of tournament {self.tournament.tourn_id}' \
                      + f' (nr. {self.tourn_nr}) is {self.tourn_stats.winner}'
        self.logs.general_log(log_message)

        self.tourn_nr += 1

        if self.tourn_nr > 0 and not self.scenario.resume:
            self.select_contenders()

        self.contender_dict = self.res_process.get_contender_dict()

        return self.rtac_data


'''
class GrayBoxTournamentManager(AbstractTournamentManager):
    """Tournament manager class for the GrayBox implementation."""

'''


def tourn_manager_factory(scenario: argparse.Namespace,
                          ta_runner: BaseTARunner, logs: RTACLogs,
                          rtac_data: RTACData) -> AbstractTournamentManager:
    """Class factory to return the initialized TournamentManager class
    appropriate to the RTAC method scenario.ac.

    :param scenario: Namespace containing all settings for the RTAC.
    :type scenario: argparse.Namespace
    :param ta_runner: Target algorithm runner object.
    :type: BaseTARunner
    :param logs: Object containing loggers and logging functions.
    :type: RTACLogs
    :returns: Inititialized TournamentManager object matching the RTAC method
        of the scenario.
    :rtype: BaseTARunner
    """
    if scenario.ac in (ACMethod.ReACTR, ACMethod.ReACTRpp):
        return TournamentManager(scenario, ta_runner, logs, rtac_data)
    elif scenario.ac is ACMethod.CPPL:
        return TournamentManagerCPPL(scenario, ta_runner, logs, rtac_data)

    '''
    elif scenario.ac is ACMethod.GRAYBOX:
        return GrayBoxTournamentManager(scenario, ta_runner, logs, rtac_data)
    '''


if __name__ == '__main__':
    pass
