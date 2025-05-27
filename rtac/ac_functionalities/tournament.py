"""This module contains classes implementing problem instance solving
tournaments by differently configured target algorithms according to the RTAC
method utilized."""

from abc import ABC, abstractmethod
import multiprocessing as mp
import subprocess
import os
import uuid
import time
import signal
from utils.process_affinity import set_affinity_recursive
from ac_functionalities.config_gens import DefaultConfigGen
from ac_functionalities.rtac_data import (
    TournamentStats,
    TARun,
    TARunStatus
)
from ac_functionalities.ta_runner import BaseTARunner
from ac_functionalities.rtac_data import Configuration, RTACData, ACMethod
from ac_functionalities.logs import RTACLogs
import argparse


class AbstractTournament(ABC):

    def __init__(self, scenario: argparse.Namespace, ta_runner: BaseTARunner,
                 rtac_data: RTACData, logs: RTACLogs) -> None:
        """Initializes tournament class for ReACTR tournaments.
        If self.scenario.baselineperf the configuration used is the default
        configuration according to the configuration space definitin json.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        :param ta_runner: Target algorithm runner object.
        :type: BaseTARunner
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        :param logs: Object containing loggers and logging functions.
        :type: RTACLogs
        """
        self.scenario = scenario
        self.ta_runner_class = ta_runner
        self.rtac_data = rtac_data
        self.logs = logs
        
        if self.scenario.baselineperf:
            self.dcg = DefaultConfigGen(self.scenario)
    
    @abstractmethod
    def start_tournament(self, instance: str,
                         contender_dict: dict[str: Configuration],
                         tourn_nr: int) -> None:
        """Sets up tournament data and information and starts the tournament
        with scenario.number_cores configured target algorithms running in
        paralel with settings according to the RTAC method used.

        :param instance: path to the problem instance to solve.
        :type instance: str
        :param contender_dict: Dictonary containing configurations to run in
            the tournament with configuration id as key and Configuration
            object as value.
        :type instance: dict[str: Configuration]
        :param tourn_nr: Number of the tournament during this RTAC run.
        :type tourn_nr: int
        """

    @abstractmethod
    def watch_tournament(self):
        """Function to observe the tournament and enforce the timelimit
        scenario.timeout if reached according to the RTAC method used."""

    def close_tournament(self, process: subprocess.Popen) -> None:
        """Function that initiates termination of all target algorithm runs.

        :param process: Target algorithm run process.
        :type process: subprocess.Popen
        """
        self.rtac_data.ev.set()
        self.rtac_data.event = 1
        print(f'\nClosing tournament Nr. {self.tourn_nr}',
              f'(Tournament ID: {self.tourn_id})',
              f'due to timeout ({self.scenario.timeout}s).\n')
        if self.scenario.objective_min:
            time.sleep(1)  # extra time for TAs to shut down and print results
        for core in range(self.scenario.number_cores):
            if self.rtac_data.status[core] not in (2, 3):
                self.rtac_data.status[core] = 5
            self.terminate_run(core, process[core])

    def terminate_run(self, core: int, process: subprocess.Popen) -> None:
        """Function that enforces termination of a target algorithm run.

        :param core: Index of the process in the list of processes.
        :type core: int
        :param process: Target algorithm run process.
        :type process: subprocess.Popen
        """
        try:
            os.kill(self.rtac_data.pids[core], signal.SIGKILL)
        except Exception as e:
            message = \
                f'Tried killing pid {self.rtac_data.pids[core]} - ' + str(e) \
                + f' - It was run with configuration {self.conf_id_list[core]}'
            self.logs.general_log(message)
        process.terminate()
        process.join()


class Tournament(AbstractTournament):
    """Tournament class with functions needed for ReACTR method tournaments."""

    def start_tournament(self, instance: str,
                         contender_dict: dict[str: Configuration],
                         tourn_nr: int) -> None:
        """Sets up tournament data and information and starts the tournament
        with scenario.number_cores configured target algorithms running in
        paralel according to the ReACTR method.

        :param instance: path to the problem instance to solve.
        :type instance: str
        :param contender_dict: Dictonary containing configurations to run in
            the tournament with configuration id as key and Configuration
            object as value.
        :type instance: dict[str: Configuration]
        :param tourn_nr: Number of the tournament during this RTAC run.
        :type tourn_nr: int
        """
        self.instance = instance
        self.tourn_nr = tourn_nr
        if self.scenario.baselineperf:
            def_conf = self.dcg.generate()
            contender_dict = {def_conf.id: def_conf}
        self.config_list = list(contender_dict.values())
        self.conf_id_list = list(contender_dict.keys())
        self.tourn_id = uuid.uuid4().hex
        log_message = f'Starting tournament {self.tourn_id}' \
                      + f' (nr. {self.tourn_nr}) on instance {self.instance}'
        self.logs.general_log(log_message)
        self.tournamentstats = \
            TournamentStats(self.tourn_id, tourn_nr, self.conf_id_list, None,
                            [], [], [], [], {})

        sync_event = mp.Event()

        for core in range(self.scenario.number_cores):
            self.ta_runner = \
                self.ta_runner_class(self.scenario, self.logs, core)
            contender = self.config_list[core]
            self.tournamentstats.TARuns[contender.id] = \
                TARun(contender.id, contender.conf, 0, 0, TARunStatus.running)

            translated_config = self.ta_runner.translate_config(contender)

            self.rtac_data.process[core] = \
                mp.Process(target=self.ta_runner.run,
                           args=[self.instance, translated_config,
                                 self.rtac_data, sync_event])

        # Starting processes
        for core in range(self.scenario.number_cores):
            self.rtac_data.process[core].start()

        sync_event.set()
        time.sleep(0.01)

        for core in range(self.scenario.number_cores):
            set_affinity_recursive(self.rtac_data.process[core], core)

    def watch_tournament(self) -> None:
        """Function to observe the tournament and enforce the timelimit
        scenario.timeout if reached."""

        while any(proc.is_alive() for proc in self.rtac_data.process):
            time.sleep(1)  # Timeout is int, so checking every second is enough
            currenttime = time.time() - self.rtac_data.start

            if currenttime >= self.scenario.timeout:
                self.close_tournament(self.rtac_data.process)


class Tournamentpp(Tournament):
    """Tournament class with functions needed for ReACTR method tournaments."""


def tournament_factory(scenario: argparse.Namespace, ta_runner: BaseTARunner,
                       rtac_data: RTACData, logs: RTACLogs) -> Tournament:
    """Class factory to return the initialized TournamentManager class
    appropriate to the RTAC method scenario.ac.

    :param scenario: Namespace containing all settings for the RTAC.
    :type scenario: argparse.Namespace
    :param ta_runner: Target algorithm runner object.
    :type: BaseTARunner
    :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
    :type rtac_data: RTACData
    :param logs: Object containing loggers and logging functions.
    :type: RTACLogs
    :returns: Inititialized Tournament object matching the RTAC method
        of the scenario.
    :rtype: Tournament
    """
    if scenario.ac in (ACMethod.ReACTR, ACMethod.CPPL):
        return Tournament(scenario, ta_runner, rtac_data, logs)
    elif scenario.ac is ACMethod.ReACTRpp:
        return Tournamentpp(scenario, ta_runner, rtac_data, logs)


if __name__ == '__main__':
    pass
