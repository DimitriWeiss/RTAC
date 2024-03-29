"""This module contains classes that guide the RTAC process according to
method."""

from abc import ABC, abstractmethod
import argparse
import sys
from ac_functionalities.ta_runner import ta_runner_factory as ta_runner
from ac_functionalities.rtac_data import rtacdata_factory as rtacdata
from ac_functionalities.tournament_manager import TournamentManager
from ac_functionalities.result_processing import ResultProcessing
from ac_functionalities.logs import RTACLogs


class AbstractRTAC(ABC):
    """Realtime Algorithm Configuration class."""

    def __init__(self, scenario: argparse.Namespace) -> None:
        """Realtime algorithm configuration class to be used to solve problem
        instances successively.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        """
        self.scenario = scenario
        if self.scenario.baselineperf:
            self.scenario.number_cores = 1
        self.logs = RTACLogs(self.scenario)
        self.logs.scenario_log(self.scenario)
        self.rtac_data = rtacdata(self.scenario)
        self.ta_runner = ta_runner
        self.tournament_manager = TournamentManager(self.scenario,
                                                    self.ta_runner,
                                                    self.logs,
                                                    self.rtac_data)
        self.result_processing = ResultProcessing(self.scenario,
                                                  self.logs)

    @abstractmethod
    def solve_instance(self, instance: str) -> None:
        """Solves problem instance and performs all associated
        functionalities."""
        ...

    @abstractmethod
    def plot_performances(self, results: bool = False,
                          times: bool = False) -> None:
        """Plots results of the logged RTAC run and saves figure."""
        ...


class RTAC(AbstractRTAC):
    """Implementation of ReACTR."""

    def solve_instance(self, instance: str) -> None:
        """Solves problem instance and performs all associated functionalities.

        :param instance: Path to the problem instance file.
        :type instance: str
        """
        if self.tournament_manager.tourn_nr > 0:
            self.rtac_data = rtacdata(self.scenario)

        self.rtac_data = self.tournament_manager.solve_instance(instance,
                                                                self.rtac_data)

        if not self.scenario.objective_min:
            if self.rtac_data.newtime >= self.scenario.timeout:
                print(f'Instance {instance} could not be solved within',
                      f'{self.scenario.timeout}s.')
            else:
                print(f'Solved instance {instance} in',
                      f'{self.rtac_data.newtime}s.')
        else:
            if self.rtac_data.best_res == sys.float_info.max * 1e-100:
                print(f'Instance {instance} could not be solved within',
                      f'{self.scenario.timeout}s.')
            else:
                print(f'Solved instance {instance} with objective value',
                      f'{self.rtac_data.best_res}.')

    def plot_performances(self, results: bool = False,
                          times: bool = False) -> None:
        """Plot results of the logged RTAC run and save figure.

        :param results: True if scenario was objective quality minimization.
        :type results: bool
        :param times: True if scenario was runtime minimization.
        :type times: bool
        """
        if results:
            ...
        elif times:
            ...
