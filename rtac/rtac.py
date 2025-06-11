"""This module contains classes that guide the RTAC process according to
method."""

from utils.background_thread_control import set_background_thread_nr

set_background_thread_nr()

from abc import ABC, abstractmethod
import argparse
import sys
import importlib
import threading
import queue
from multiprocessing.sharedctypes import Synchronized
import multiprocessing as mp
from ac_functionalities.ta_runner import ta_runner_factory as ta_runner
from ac_functionalities.rtac_data import rtacdata_factory as rtacdata, ACMethod
from ac_functionalities.tournament_manager import tourn_manager_factory as TM
from ac_functionalities.logs import RTACLogs


class AbstractRTAC(ABC):
    """Realtime Algorithm Configuration class."""

    def __init__(self, scenario: argparse.Namespace) -> None:
        """Realtime algorithm configuration class to be used to solve problem
        instances successively.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        """
        self.huge_float = sys.float_info.max * 1e-100
        self.scenario = scenario
        if self.scenario.baselineperf:
            self.scenario.number_cores = 1
        self.logs = RTACLogs(self.scenario)
        self.logs.scenario_log(self.scenario)
        self.ta_runner = ta_runner
        self.init_tournament_manager()

    def init_tournament_manager(self) -> None:
        self.rtac_data = rtacdata(self.scenario)
        self.tournament_manager = TM(self.scenario, self.ta_runner, self.logs,
                                     self.rtac_data)

    def init_rtac_data(self) -> None:
        """Initializes new RTAC data."""
        if self.tournament_manager.tourn_nr > 0:
            self.rtac_data = rtacdata(self.scenario)

    @abstractmethod
    def solve_instance(self, instance: str) -> None:
        """Solves problem instance and performs all associated
        functionalities."""

    @abstractmethod
    def plot_performances(self, results: bool = False,
                          times: bool = False) -> None:
        """Plots results of the logged RTAC run and saves figure."""


class RTAC(AbstractRTAC):
    """Implementation of ReACTR."""

    def solve_instance(self, instance: str, next_instance: str = None,
                       early_rtac_data=None) -> None:
        if self.scenario.gray_box:
            self.gray_box(instance, next_instance, early_rtac_data)
        else:
            self.black_box(instance)

    def black_box(self, instance: str = None) -> None:
        """Solves problem instance and performs all associated functionalities.

        :param instance: Path to the problem instance file.
        :type instance: str
        """
        self.init_rtac_data()

        print('\n\n')

        print(f'Starting instance {instance}...')

        self.rtac_data = self.tournament_manager.solve_instance(instance,
                                                                self.rtac_data,
                                                                self)

        self.result_output(instance)

    def gray_box(self, instance: str, next_instance: str = None,
                 early_rtac_data=None) -> None:
        """Solves problem instance and performs all associated functionalities.

        :param instance: Path to the problem instance file.
        :type instance: str
        """
        self.init_rtac_data()

        self.instance = instance
        self.early_instance = mp.Manager().list()
        if next_instance:
            self.early_instance = [next_instance]

        print('\n\n')

        if early_rtac_data is not None:
            print(f'Starting instance {instance} with time advantage...')
            rtac_data = early_rtac_data
        else:
            print(f'Starting instance {instance}...')
            rtac_data = self.rtac_data

        self.rtac_thread = threading.Thread(
            target=self.tournament_manager.solve_instance,
            args=(instance, rtac_data),
            kwargs={'rtac': self, 'next_instance': self.early_instance})
        self.rtac_thread.start()

    def provide_early_instance(self, early_instance):
        self.early_instance.append(early_instance)

    def wrap_up_gb(self):
        self.rtac_thread.join()
        self.rtac_data = self.tournament_manager.rtac_data
        self.result_output(self.instance)

    def result_output(self, instance):
        if not self.rtac_data.skip:

            print('\n')

            if not self.scenario.objective_min:
                if isinstance(self.rtac_data.newtime, Synchronized):
                    newtime = self.rtac_data.newtime.value
                else:
                    newtime = self.rtac_data.newtime
                if newtime >= self.scenario.timeout:
                    print(f'Instance {instance} could not be solved within',
                          f'{self.scenario.timeout}s.')
                else:
                    print(f'Solved instance {instance} in',
                          f'{self.rtac_data.newtime}s.')
            else:
                if self.rtac_data.best_res == self.huge_float:
                    print(f'Instance {instance} could not be solved within',
                          f'{self.scenario.timeout}s.')
                else:
                    print(f'Solved instance {instance} with objective value',
                          f'{self.rtac_data.best_res}.')

            print('.\n' * 3)

        else:
            print('\n' * 2)
            print('.\n' * 3)
            print('* Instance', self.instance,
                  'already solved in previous tournament.')
            print('\n')
            print('* Continuing')
            print('.\n' * 3)

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


class RTACpp(RTAC):
    """Implementation of ReACTR++."""

    def init_tournament_manager(self) -> None:
        module = importlib.import_module(self.scenario.wrapper)
        name = self.scenario.wrapper_name
        wrapper = getattr(module, name)()
        self.interim_meaning = wrapper.interim_info()
        self.rtac_data = \
            rtacdata(self.scenario, interim_meaning=self.interim_meaning)
        self.tournament_manager = TM(self.scenario, self.ta_runner, self.logs,
                                     self.rtac_data)

    def init_rtac_data(self) -> None:
        """Initializes new RTAC data. Override for ReACTR++ implementation."""
        if self.tournament_manager.tourn_nr > 0:
            self.rtac_data = \
                rtacdata(self.scenario, interim_meaning=self.interim_meaning)


def rtac_factory(scenario: argparse.Namespace) -> AbstractRTAC:
    """Class factory to return the initialized RTAC class
    appropriate to the RTAC method scenario.ac.

    :param scenario: Namespace containing all settings for the RTAC.
    :type scenario: argparse.Namespace
    :returns: Inititialized AbstractRTAC object matching the RTAC method
        of the scenario.
    :rtype: AbstractRTAC
    """
    if scenario.ac in (ACMethod.ReACTR, ACMethod.CPPL):
        return RTAC(scenario)
    elif scenario.ac is ACMethod.ReACTRpp:
        return RTACpp(scenario)


if __name__ == '__main__':
    pass
