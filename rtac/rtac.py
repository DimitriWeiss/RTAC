"""This module contains classes that guide the RTAC process according to
method."""

from rtac.utils.background_thread_control import set_background_thread_nr

set_background_thread_nr()

from abc import ABC, abstractmethod
import argparse
import sys
import importlib
import threading
from multiprocessing.sharedctypes import Synchronized
import multiprocessing as mp
from rtac.ac_functionalities.ta_runner import ta_runner_factory as ta_runner
from rtac.ac_functionalities.rtac_data import rtacdata_factory as rtacdata, ACMethod, RTACData, RTACDatapp
from rtac.ac_functionalities.tournament_manager import tourn_manager_factory as TM
from rtac.ac_functionalities.logs import RTACLogs
import faulthandler
faulthandler.enable()


class AbstractRTAC(ABC):
    """
    Realtime Algorithm Configuration class.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    """

    def __init__(self, scenario: argparse.Namespace):
        """
        Realtime algorithm configuration class to be used to solve problem
        instances successively.
        """
        self.huge_float = sys.float_info.max * 1e-100
        self.scenario = scenario
        if self.scenario.baselineperf:
            self.scenario.number_cores = 1
        self.logs = RTACLogs(self.scenario)
        self.logs.scenario_log(self.scenario)
        self.ta_runner = ta_runner
        self.init_tournament_manager()
        self.es = False
        self.early_instance = mp.Manager().list([None])

    def init_tournament_manager(self) -> None:
        """
        Initialize tournament manager object.

        Returns
        -------
        None
        """
        self.rtac_data = rtacdata(self.scenario)
        self.tournament_manager = TM(self.scenario, self.ta_runner, self.logs,
                                     self.rtac_data)

    def init_rtac_data(self) -> None:
        """
        Initialize new RTAC data.

        Returns
        -------
        None
        """
        if self.tournament_manager.tourn_nr > 0:
            self.rtac_data = rtacdata(self.scenario)

    @abstractmethod
    def solve_instance(self, instance: str) -> None:
        """
        Solves problem instance and performs all associated
        functionalities.

        Parameters
        ----------
        instance : str
            Path to problem instance.

        Returns
        -------
        None
        """


class RTAC(AbstractRTAC):
    """
    Implementation of ReACTR.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    """

    def __init__(self, scenario: argparse.Namespace):
        """
        Initialize RTAC object.
        """
        AbstractRTAC.__init__(self, scenario)
        self.rtac_thread = [None, None]

    def solve_instance(self, instance: str, next_instance: str = None,
                       early_rtac_data: RTACData | RTACDatapp = None) -> None:
        """
        Solve problem instance.

        Parameters
        ----------
        instance : str
            Path to problem instance.
        next_instance : str
            Path to next problem instance. Defaults to None.
        early_rtac_data : RTACData | RTACDatapp
            Object containing data and objects necessary throughout the RTAC 
            modules.

        Returns
        -------
        None
        """
        if self.scenario.gray_box:
            self.gray_box(instance, next_instance, early_rtac_data)
        else:
            self.black_box(instance)

    def black_box(self, instance: str = None) -> None:
        """
        Solves problem instance and performs all associated functionalities.

        Parameters
        ----------
        instance : str
            Path to problem instance. Defaults to None.

        Returns
        -------
        None
        """
        self.init_rtac_data()

        print('\n\n')

        self.rtac_data = self.tournament_manager.solve_instance(instance,
                                                                self.rtac_data)

        self.result_output(instance)

    def gray_box(self, instance: str, next_instance: str = None,
                 early_rtac_data: RTACData | RTACDatapp = None) -> None:
        """
        Solves problem instance and performs all associated functionalities.

        Parameters
        ----------
        instance : str
            Path to problem instance.
        next_instance : str
            Path to next problem instance. Defaults to None.
        early_rtac_data : RTACData | RTACDatapp
            Object containing data and objects necessary throughout the RTAC 
            modules.

        Returns
        -------
        None
        """
        if instance != self.early_instance[0]:
            self.instance = instance
        elif self.early_instance[0] is not None:
            self.es_instance = instance
        
        if next_instance:
            self.early_instance = [next_instance]

        print('\n\n')

        if early_rtac_data is not None:
            self.es = True
            es_rtac_data = early_rtac_data
            thread_idx = 1
        else:
            self.init_rtac_data()
            es_rtac_data = None
            thread_idx = 0

        self.rtac_thread[thread_idx] = threading.Thread(
            target=self.tournament_manager.solve_instance,
            args=(instance, self.rtac_data),
            kwargs={'rtac': self, 'next_instance': self.early_instance,
                    'es_rtac_data': es_rtac_data})
        self.rtac_thread[thread_idx].start()

    def provide_early_instance(self, early_instance: str) -> None:
        """
        If next instance is available early, use this function to pass it to 
        the gray-box RAC method.

        Parameters
        ----------
        early_instance : str
            Path to next problem instance. Defaults to None.

        Returns
        -------
        None
        """
        self.early_instance[0] = early_instance

    def wrap_up_gb(self) -> None:
        """
        Wrap up gray-box RAC method for this ionstance / instance-pair.

        Returns
        -------
        None
        """
        for rtac_thread in self.rtac_thread:
            if isinstance(rtac_thread, threading.Thread):
                rtac_thread.join()
        self.rtac_data = self.tournament_manager.rtac_data
        self.result_output(self.instance)
        self.early_instance = mp.Manager().list([None])

    def result_output(self, instance: str) -> None:
        """
        Print result to terminal.

        Parameters
        ----------
        instance : str
            Path to problem instance.

        Returns
        -------
        None
        """
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

    def plot_performances(self, results: bool = False,
                          times: bool = False) -> None:
        """
        Plots results of the logged RTAC run and saves figure.

        Parameters
        ----------
        results : bool
            True if objective value minimization scenario. Defaults to False.
        times : bool
            True if runtime minimization scenario. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            This method is not implemented yet.
        """

        raise NotImplementedError("This method is not implemented yet.")


class RTACpp(RTAC):
    """
    Implementation of ReACTR++.
    """

    def init_tournament_manager(self) -> None:
        """
        Initialize tournament manager object for ReACTR++.

        Returns
        -------
        None
        """
        module = importlib.import_module(self.scenario.wrapper)
        name = self.scenario.wrapper_name
        wrapper = getattr(module, name)()
        self.interim_meaning = wrapper.interim_info()
        self.rtac_data = \
            rtacdata(self.scenario, interim_meaning=self.interim_meaning)
        self.tournament_manager = TM(self.scenario, self.ta_runner, self.logs,
                                     self.rtac_data)

    def init_rtac_data(self) -> None:
        """
        Initialize new RTAC data for ReACTR++.

        Returns
        -------
        None
        """
        if self.tournament_manager.tourn_nr > 0:
            self.rtac_data = \
                rtacdata(self.scenario, interim_meaning=self.interim_meaning)


def rtac_factory(scenario: argparse.Namespace) -> AbstractRTAC:
    """
    Class factory to return the initialized RTAC class
    appropriate to the RTAC method `scenario.ac`.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.

    Returns
    -------
    AbstractRTAC
        Initialized AbstractRTAC object matching the RTAC method of the 
        scenario.
    """
    if scenario.ac in (ACMethod.ReACTR, ACMethod.CPPL):
        return RTAC(scenario)
    elif scenario.ac is ACMethod.ReACTRpp:
        return RTACpp(scenario)


if __name__ == '__main__':
    pass
