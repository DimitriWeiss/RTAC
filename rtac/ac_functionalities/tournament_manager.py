"""This module contans classes for touenament management according to the RTAC
method used."""

from abc import ABC, abstractmethod
import argparse
import threading
import time
import copy
import gc
import sys
from utils.clean_logs import remove_fuse_hidden_files as rfhf
from collections import OrderedDict
from multiprocessing.sharedctypes import Synchronized
from ac_functionalities.tournament import tournament_factory
from ac_functionalities.result_processing import processing_factory
from ac_functionalities.rtac_data import TARunStatus, RTACData, ACMethod
from ac_functionalities.ta_runner import BaseTARunner
from ac_functionalities.logs import RTACLogs
from ac_functionalities.ranking.gray_box import Gray_Box


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
        self.huge_float = sys.float_info.max * 1e-100
        self.scenario = scenario
        self.ta_runner = ta_runner
        self.rtac_data = rtac_data
        self.logs = logs
        self.instance_history = []

        self.tournament = tournament_factory(self.scenario, self.ta_runner,
                                             self.rtac_data, self.logs)

        self.res_process = processing_factory(self.scenario, self.logs)

        if self.scenario.resume:
            print('\n')
            print('Resuming from previous run.')
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
            print('\n')
            print('Running in experimental mode.')
            self.tourn_nr = 0
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

        else:
            self.tourn_nr = 0
            self.contender_dict = self.res_process.get_contender_dict()

        if self.scenario.gray_box:
            self.es_tourn_nr = 0
            #self.tourn_nr_list = [self.tourn_nr]

        self.tourn_nr_list = [self.tourn_nr]

        self.logs.init_rtac_logs()
        self.logs.init_ranking_logs()

    def set_tourn_status(self, tournamentstats, rtac_data, tournament) -> None:
        """Setting the results of the tournament and status of the target
        algorithm runs to rtac_data.TournamentStats."""
        tournamentstats.results = rtac_data.ta_res[:]
        tournamentstats.times = rtac_data.ta_res_time[:]
        tournamentstats.rtac_times = rtac_data.ta_rtac_time[:]
        if self.scenario.verbosity == 2:
            print('* Tournament:', tournament.tourn_id, 'consisted of',
                  len(tournamentstats.TARuns), 'contenders.')
        for tr, tarun in enumerate(tournamentstats.TARuns):
            tournamentstats.TARuns[tarun].res = rtac_data.ta_res[tr]
            tournamentstats.TARuns[tarun].time = \
                rtac_data.ta_res_time[tr]
            tournamentstats.TARuns[tarun].status = \
                TARunStatus(rtac_data.status[tr])

        return tournamentstats

    @abstractmethod
    def solve_instance(self, instance: str, rtac_data: RTACData,
                       next_instance: str = None, rtac=None) -> RTACData:
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

    def manage_tournament(self, instance: str, rtac_data: RTACData):

        msg = f'#  Start solving instance {instance}...  #'
        len_msg = len(msg)
        print('#' * len_msg)
        print('#' + ' ' * (len_msg - 2) + '#')
        print(msg)
        print('#' + ' ' * (len_msg - 2) + '#')
        print('#' * len_msg)

        self.instance = instance
        cores_start = [i for i in range(self.scenario.number_cores)]
        self.tournament.start_tournament(self.instance, self.contender_dict,
                                         self.tourn_nr, cores_start)
        # self.rtac_data = self.tournament.rtac_data = rtac_data
        self.tournament.watch_tournament()

        # Update tournament status
        self.tournamentstats = self.tournament.tournamentstats
        self.tournamentstats.winner = self.rtac_data.winner.value
        self.res_process.process_tourn(self.rtac_data, self.instance)

        self.general_logging(scenario=self.scenario,
                             rtac_data=rtac_data,
                             tournamentstats=self.tournamentstats,
                             tourn_nr=self.tourn_nr,
                             tournament=self.tournament,
                             instance=instance)
        
        self.contender_dict = self.res_process.get_contender_dict()

        self.instance_history.append(instance)

        # self.tourn_nr += 1
        self.tourn_nr = self.tourn_nr_list[-1] + 1
        self.tourn_nr_list.append(self.tourn_nr)

        self.res_process.tourn_nr = self.tourn_nr
        self.tournament.tourn_nr = self.tourn_nr

    def general_logging(self, scenario=None, rtac_data=None,
                        tournamentstats=None, tourn_nr=None,
                        tournament=None, early_tourn=False,
                        instance=None) -> None:
        if scenario is None:
            scenario = self.scenario
            rtac_data = self.rtac_data
            tournamentstats = self.tournamentstats
            tourn_nr = self.tourn_nr
            tournament = self.tournament
        print('\n')
        if self.scenario.verbosity == 2:
            if early_tourn:
                message = '-' * 157 + '\n' \
                          + 'Instance ' + str(instance) \
                          + ' solved in early starting tournament ' \
                          + str(tournament.tourn_id) + '.\n\n' \
                          + 'Unadjusted time results including time' \
                          + ' advantage are:' + '\n\n' \
                          + 'Runtimes stated by TA: ' \
                          + str(rtac_data.ta_res_time[:]) + '\n' \
                          + 'Runtimes with TA runner: ' \
                          + str(rtac_data.ta_rtac_time[:]) + '\n\n' \
                          + '* Cancelled configurations are set to timeout.' \
                          + '\n\n'
                message += '* The following time results are adjusted to not' \
                           + ' include the extra runtime.\n\n\n'
                message += '- ' * 78 + '\n'

                rtac_data = self.adjust_time_results(rtac_data)

            else:
                message = 'Instance ' + str(instance) \
                          + ' solved in tournament ' \
                          + str(tournament.tourn_id) + '.\n\n'
            message += 'Objective values: ' + str(rtac_data.ta_res[:]) \
                + '\n\n' \
                + '* Cancelled configurations are set to Big M.' + '\n\n\n' \
                + 'Runtimes stated by TA: ' \
                + str(rtac_data.ta_res_time[:]) + '\n' \
                + 'Runtimes with TA runner: ' \
                + str(rtac_data.ta_rtac_time[:]) + '\n\n' \
                + '* Cancelled configurations are set to timeout.\n'
            if early_tourn:
                message += '-' * 157
            message += '\n'
            print(message)
            self.logs.general_log(message)
        tournamentstats = self.set_tourn_status(tournamentstats,
                                                rtac_data,
                                                tournament)
        if self.scenario.verbosity != 2 and early_tourn:
            rtac_data = self.adjust_time_results(rtac_data)
        self.logs.rtac_log(rtac_data, tournamentstats)
        log_message = \
            f'Winner of tournament {tournament.tourn_id}' \
            + f' (nr. {tourn_nr}) is {tournamentstats.winner}'
        self.logs.general_log(log_message)

    def adjust_time_results(self, rtac_data):
        if all(res == self.scenario.timeout
                for res in rtac_data.ta_res_time[:]):
            self.finished.wait()
        for i, res in enumerate(rtac_data.ta_res_time):
            if res != float(self.scenario.timeout) and i in self.term_list:
                rtac_data.ta_res_time[i] = \
                    max(0,
                        round(res - min(
                            self.rtac_data.ta_res_time) + self.currenttime, 2
                        ))
        for i, res in enumerate(rtac_data.ta_rtac_time):
            if res != float(self.scenario.timeout) and i in self.term_list:
                rtac_data.ta_rtac_time[i] = \
                    max(0,
                        round(res - min(
                            self.rtac_data.ta_rtac_time) + self.currenttime, 2
                        ))

        return rtac_data

    def get_tourn_nr(self, rtac_data):
        if not bool(rtac_data.early_start_tournament.value):
            tourn_nr = self.tourn_nr
        else:
            tourn_nr = self.es_tourn_nr

        return tourn_nr


class TournamentManager(AbstractTournamentManager):
    """Tournament manager class for the ReACTR implementation."""

    def solve_instance(self, instance: str, rtac_data: RTACData,
                       **kwargs) -> RTACData:
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

        self.rtac_data = self.tournament.rtac_data = rtac_data
        tourn_nr = self.get_tourn_nr(rtac_data)

        self.kwargs = kwargs
        self.logs.ranking_log(self.res_process.pool, self.res_process.scores,
                              tourn_nr, self.contender_dict)

        self.manage_tournament(instance, rtac_data)

        return self.rtac_data


class TournamentManagerCPPL(AbstractTournamentManager):
    """Tournament manager class for the CPPL implementation."""

    def solve_instance(self, instance: str, rtac_data: RTACData,
                       **kwargs) -> RTACData:
        """Solving the problem instance according to the CPPL implementation.
        :param instance: path to the problem instance to solve.

        :type instance: str
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        :returns: Updated object containing data and objects necessary
            throughout the rtac modules
        :rtype: RTACData
        """

        self.rtac_data = self.tournament.rtac_data = rtac_data
        tourn_nr = self.get_tourn_nr(rtac_data)

        self.kwargs = kwargs
        self.logs.ranking_log(self.res_process.pool, self.res_process.bandit,
                              tourn_nr, self.contender_dict,
                              bandit_models=self.res_process.bandit_models)

        self.manage_tournament(instance, rtac_data)

        return self.rtac_data


class GrayBox:
    """This class contains functions needed for the gray-box
    functionality of the tournament manager classes."""

    def train_gray_box_model(self):

        self.gb_pw_inst_archive = self.tournament.gb_pw_inst_archive

        if self.gb_pw_inst_archive:

            cores, s_instances = self.tournament.pw_cores, \
                self.tournament.s_instances

            if not self.scenario.objective_min:
                res = list(self.rtac_data.ta_res_time)
            else:
                res = list(self.rtac_data.ta_res)

            winner = res.index(min(res))

            self.X_train, self.y_train, self.cost_mat_train = \
                self.gray_box.prepare_train_data(
                    self.X_train, self.gb_pw_inst_archive, cores, winner, res,
                    s_instances, self.y_train, self.cost_mat_train
                )

            self.gb_model = \
                self.gray_box.train_gb(self.X_train, self.y_train,
                                       self.cost_mat_train,
                                       self.scenario.number_cores)

            # Pass TournamentManager object reference to Tournament object
            # so it can call early_start, etc.
            self.tournament.tm = self
            self.tournament.gray_box = self.gray_box
            self.tournament.gb_model = self.gb_model

    def manage_tournament(self, instance: str, rtac_data: RTACData, **kwargs):
        if instance not in self.instance_history:
            self.instance_history.append(instance)
            if self.kwargs['es_rtac_data'] is None:
                msg = f'#  Start solving instance {instance}...  #'
                len_msg = len(msg)
                print('#' * len_msg)
                print('#' + ' ' * (len_msg - 2) + '#')
                print(msg)
                print('#' + ' ' * (len_msg - 2) + '#')
                print('#' * len_msg)

                self.rtac_data = rtac_data

                cores_start = [i for i in range(self.scenario.number_cores)]
                self.finished = threading.Event()
                self.rtac = self.kwargs['rtac']
                self.next_instance = \
                    self.kwargs['next_instance'][0] \
                    if self.kwargs['next_instance'] \
                    else None
                self.gray_box = Gray_Box()
                self.tournament.gray_box = self.gray_box
                self.tournament.tm = self
                self.X_train = []
                self.y_train, self.cost_mat_train = [], []
                self.tournament.terminated_configs = []
                self.instance = instance
                self.rtac_data = self.tournament.rtac_data = rtac_data
                self.tournament.start_tournament(self.instance,
                                                 self.contender_dict,
                                                 self.tourn_nr, cores_start)
                self.tournament.watch_tournament()
                self.tournamentstats = self.tournament.tournamentstats
                self.tournamentstats.winner = self.rtac_data.winner.value

                # Only process tournament if it was a regular one
                self.res_process.process_tourn(self.rtac_data, self.instance,
                                               self.tourn_nr)
                self.general_logging(instance=self.instance)
                self.contender_dict = self.res_process.get_contender_dict()
                self.train_gray_box_model()

                self.tourn_nr = self.tourn_nr_list[-1] + 1
                self.tourn_nr_list.append(self.tourn_nr)
                self.res_process.tourn_nr = self.tourn_nr
                self.tournament.tourn_nr = self.tourn_nr

                self.finished.set()
                gc.collect
            else:
                msg = f'#  Start solving instance {instance} ' + \
                    'with time advantage...  #'
                len_msg = len(msg)
                print('#' * len_msg)
                print('#' + ' ' * (len_msg - 2) + '#')
                print(msg)
                print('#' + ' ' * (len_msg - 2) + '#')
                print('#' * len_msg)

                self.es_rtac_data = self.kwargs['es_rtac_data']

                cores_start = \
                    [core for core in range(self.scenario.number_cores)
                     if core in self.term_list]
                self.next_instance = instance
                self.es_scenario = copy.deepcopy(self.scenario)
                self.es_scenario.timeout = \
                    self.scenario.timeout + self.time_advantage
                self.es_tournament = tournament_factory(self.es_scenario,
                                                        self.ta_runner,
                                                        rtac_data,
                                                        self.logs)
                self.es_tournament.rtac_data = self.es_rtac_data
                contender_dict = self.res_process.get_contender_dict()

                # Put best contender at indices of term_list
                items = list(contender_dict.items())
                num_to_move = len(self.term_list)
                moved_items = items[:num_to_move]
                remaining_items = items[num_to_move:]
                for i, idx in enumerate(sorted(self.term_list)):
                    remaining_items.insert(idx, moved_items[i])
                contender_dict = OrderedDict(remaining_items)

                self.es_tournament.start_tournament(self.next_instance,
                                                    contender_dict,
                                                    self.es_tourn_nr,
                                                    cores_start)
                self.es_tournament.tourn_nr = self.es_tourn_nr
                
                self.es_tournament.watch_tournament(early_tournament=True)
                self.es_tournamentstats = self.es_tournament.tournamentstats
                self.es_tournamentstats.winner = self.es_rtac_data.winner.value
                self.general_logging(scenario=self.es_scenario,
                                     rtac_data=self.es_rtac_data,
                                     tournamentstats=self.es_tournamentstats,
                                     tourn_nr=self.es_tourn_nr,
                                     tournament=self.es_tournament,
                                     early_tourn=True,
                                     instance=self.next_instance)
                self.early_rtac_data.newtime = \
                    min(self.es_rtac_data.ta_res_time[:])
                results = []
                if self.scenario.objective_min:
                    for core in range(self.scenario.number_cores):
                        results.append(self.es_rtac_data.ta_res[core])
                else:
                    for core in range(self.scenario.number_cores):
                        results.append(self.es_rtac_data.ta_res_time[core])

                self.res_process.\
                    result_summary_terminal(results, self.es_tourn_nr)

                self.es_output(self.next_instance)

                self.es_rtac_data.early_start_tournament.value = False
                self.rtac_data.early_start_tournament.value = False

                self.early_finished.set()
                gc.collect
               
            self.rtac_data.skip = False

            rfhf(self.scenario.log_folder)
                
        else:
            # Skipping problem instance if it was already solved in an early
            # starting tournment
            self.rtac_data.skip = True

    def early_start(self, currenttime):
        self.term_list = copy.deepcopy(self.tournament.term_list)
        if self.next_instance:
            print('\n')
            print('Next problem instance was provided!\n')
            self.currenttime = currenttime
            self.early_finished = threading.Event()
            for c in self.term_list:
                self.rtac_data.status[c] = 4  # TARunStatus.terminated
                self.tournament.terminate_run(c, self.rtac_data.process[c])
            self.time_advantage = \
                int(self.scenario.timeout - int(time.time() - (
                    max(self.tournament.rtac_data.substart_wall)
                )))
            self.tournament.terminated_configs = copy.deepcopy(self.term_list)
            self.early_rtac_data = self.rtac_data.early_rtac_copy()
            for c in range(self.scenario.number_cores):
                if c not in self.term_list:
                    # TARunStatus.awaiting_start -> 6
                    self.early_rtac_data.status[c] = 6
            self.early_rtac_data.early_start_tournament.value = True
            self.rtac_data.early_start_tournament.value = True
            self.early_rtac_data.cores_start = copy.deepcopy(self.term_list)
            self.es_tourn_nr = self.tourn_nr_list[-1] + 1
            self.tourn_nr_list.append(self.es_tourn_nr)
            print('\n')
            print(f'Letting contenders start early at {currenttime}s',
                  'of runtime of previous tournament on freed cores.')
            self.rtac.solve_instance(self.next_instance,
                                     None, self.early_rtac_data)

            threading.Thread(target=self.wait_for_event, daemon=True).start()

    def wait_for_event(self):
        self.finished.wait()
        self.fill_early_tournament()

    def fill_early_tournament(self):
        if not self.early_finished.is_set():
            self.es_tournament.contender_dict = \
                self.res_process.get_contender_dict()
            self.early_rtac_data.cores_start = \
                [core for core in range(self.scenario.number_cores)
                 if core not in self.term_list]
            self.es_tournament.config_list = \
                [list(self.es_tournament.contender_dict.values())[i]
                 if i in self.early_rtac_data.cores_start
                 else self.es_tournament.config_list[i]
                 for i in range(self.scenario.number_cores)]
            self.es_tournament.conf_id_list = \
                [list(self.es_tournament.contender_dict.keys())[i]
                 if i in self.early_rtac_data.cores_start
                 else self.es_tournament.config_list[i].id
                 for i in range(self.scenario.number_cores)]
            self.es_tournament.scenario.timeout = self.scenario.timeout
            self.es_tournament.fill_tournament(
                self.early_rtac_data.cores_start
            )

    def es_output(self, instance):
        print('\n')

        if not self.es_scenario.objective_min:
            if isinstance(self.early_rtac_data.newtime, Synchronized):
                newtime = self.early_rtac_data.newtime.value
            else:
                newtime = self.early_rtac_data.newtime
            if newtime >= self.es_scenario.timeout:
                print(f'Instance {instance} could not be solved within',
                      f'{self.es_scenario.timeout}s.')
            else:
                print(f'Solved instance {instance} in',
                      f'{self.early_rtac_data.newtime}s.')
        else:
            if self.early_rtac_data.best_res == self.huge_float:
                print(f'Instance {instance} could not be solved within',
                      f'{self.es_scenario.timeout}s.')
            else:
                print(f'Solved instance {instance} with objective value',
                      f'{self.early_rtac_data.best_res}.')

        print('.\n' * 3)


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
        tourn_manager = TournamentManager
    elif scenario.ac is ACMethod.CPPL:
        tourn_manager = TournamentManagerCPPL

    if scenario.gray_box:
        for name, func in GrayBox.__dict__.items():
            if callable(func) and not name.startswith("__"):
                setattr(tourn_manager, name, func)

    return tourn_manager(scenario, ta_runner, logs, rtac_data)


if __name__ == '__main__':
    pass
