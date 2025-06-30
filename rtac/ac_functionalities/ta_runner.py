"""In this module classes to run the target algorithm and observe its output
through the wrapper are implemented."""

from abc import ABC, abstractmethod
from typing import Any
from multiprocessing import Event
import fcntl
import os
import sys
import time
import signal
import importlib
import argparse
from rtac.ac_functionalities.rtac_data import (
    ACMethod,
    Configuration,
    RTACData,
    RTACDatapp
)
from rtac.ac_functionalities.logs import RTACLogs


def non_block_read(ta_output: bytes, logs: RTACLogs = None) -> str:
    """
    Function for reading `subprocess.PIPE` output without blocking
    the application until there is output. It checks for output and exits
    immediately if none is available.

    Parameters
    ----------
    ta_output : bytes
        `subprocess.PIPE` output to be read.
    logs : RTACLogs
        Object containing loggers and logging functions. Defaults to None.

    Returns
    -------
    str
        Either the output as a string or an empty string if no output was 
        available.
    """
    fd = ta_output.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        return ta_output.readline()
    except Exception as e:
        if logs is not None:
            logs.general_log(e)
        return ''


class AbstractTARunner(ABC):
    """
    Abstract TARunner.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    logs : RTACLogs
        Object containing loggers and logging functions.
    core : int
        Number of the parallel run started on a core.
    """

    @abstractmethod
    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs,
                 core: int):
        """
        Initializes target algorithm runner, loads and instantiates target
        algorithm wrapper.
        """

    @abstractmethod
    def translate_config(self, config: Configuration) -> Any:
        """
        Convert dictionary representation of the configuration to the format
        needed by the wrapper to pass to the target algorithm.

        Parameters
        ----------
        config : Configuration
            Configuration of parameter values to run the problem instance with.

        Returns
        -------
        Any
            New representation of the configuration suitable for the target 
            algorithm.
        """

    @abstractmethod
    def start_run(self, instance: str, config: Configuration,
                  rtac_data: RTACData | RTACDatapp) -> None:
        """
        Starts the target algorithm run and populates data necessary for
        coronation of tournament members.

        Parameters
        ----------
        instance : str
            Path to the problem instance to solve.
        config : Configuration
            Representation of the configuration.
        rtac_data : RTACData | RTACDatapp
            Object containing data and objects necessary throughout the RTAC 
            modules.

        Returns
        -------
        None
        """

    @abstractmethod
    def check_output(self, ta_output: bytes) -> None:
        """
        Checks the output, if there was any, and declares the instance as 
        solved by the contender if the corresponding marker is present.

        Parameters
        ----------
        ta_output : bytes
            Output from `subprocess.PIPE`.

        Returns
        -------
        None
        """

    @abstractmethod
    def check_result(self) -> None:
        """
        If this contender solved the problem instance the rtac data is
        populated by the resulting information according to the RTAC method
        scenario.ac.

        Returns
        -------
        None
        """

    @abstractmethod
    def kill_run(self) -> None:
        """
        Terminates this process/ target algorithm run, as well as the other
        contenders. Several layers of termination are included to ensure
        termination on different platforms.

        Returns
        -------
        None
        """

    @abstractmethod
    def run(self, instance: str, config: Configuration,
            rtac_data: RTACData | RTACDatapp) -> None:
        """
        Manages the target algorithm runner functions depending on the state of
        the run, according to the RTAC method specified in `scenario.ac`.

        Parameters
        ----------
        instance : str
            Path to the problem instance to solve.
        config : Configuration
            Configuration of parameter values to run the problem instance with.
        rtac_data : RTACData | RTACDatapp
            Object containing data and objects necessary throughout the RTAC 
            modules.

        Returns
        -------
        None
        """


class BaseTARunner(AbstractTARunner):
    """
    Target algorithm runner for ReACTR implementation.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    logs : RTACLogs
        Object containing loggers and logging functions.
    core : int
        Identifier for the parallel run on a specific core.
    """
    
    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs,
                 core: int) -> None:
        """
        Initializes the target algorithm runner, loads, and instantiates the
        target algorithm wrapper.
        """
        self.scenario = scenario
        if self.scenario.gray_box:
            self.last_check = time.time()
            self.clock_start = time.time()
        self.logs = logs
        self.core = core
        self.max_float = sys.float_info.max * 1e-100
        module = importlib.import_module(scenario.wrapper)
        name = scenario.wrapper_name
        self.wrapper = getattr(module, name)()

    def translate_config(self, config: Configuration) -> Any:
        """
        Convert dictionary representation of the configuration to the format
        needed by the wrapper to pass to the target algorithm.

        Parameters
        ----------
        config : Configuration
            Configuration of parameter values to run the problem instance with.

        Returns
        -------
        Any
            New representation of the configuration.
        """
        self.config_id = config.id
        translated_config = self.wrapper.translate_config(config)

        return translated_config

    def start_run(self, instance: str, config: Any,
                  rtac_data: RTACData | RTACDatapp) -> None:
        """
        Starts the target algorithm run and populates data necessary for
        coronation of tournament members.

        Parameters
        ----------
        instance : str
            Path to the problem instance to solve.
        config : Any
            Representation of the configuration.
        rtac_data : RTACData | RTACDatapp
            Object containing data and objects necessary throughout the RTAC 
            modules.

        Returns
        -------
        None
        """
        self.config = config
        self.instance = instance
        self.rtac_data = rtac_data

        self.rtac_data.substart[self.core] = time.process_time_ns()
        self.rtac_data.substart_wall[self.core] = time.time()
        if self.scenario.objective_min:
            self.om_start = time.time()

        self.proc, self.proc_cpu_time = \
            self.wrapper.start(self.config, self.scenario.timeout,
                               self.instance)
        
        self.pid = self.proc.pid
        self.rtac_data.pids[self.core] = self.pid
        self.running = True
        self.rtac_data.status[self.core] = 1  # TARunStatus.running

    def check_output(self, ta_output: bytes) -> None:
        """
        Checks the output, if there was any, and declares the instance as 
        solved by the contender if the corresponding marker is present.

        Parameters
        ----------
        ta_output : bytes
            Output from `subprocess.PIPE`.

        Returns
        -------
        None
        """
        if ta_output != b'':
            if_solved = \
                self.wrapper.check_if_solved(ta_output, non_block_read,
                                             self.proc)

            if if_solved is not None:
                res, self.time, self.rtac_data.event = if_solved
                self.rtac_data.ta_res[self.core] = res
                self.rtac_data.ta_res_time[self.core] = self.time
                self.rtac_data.status[self.core] = 2  # TARunStatus.finished

            if self.scenario.gray_box:
                self.gb_record(ta_output)

    def check_result(self) -> None:
        """
        If this contender solved the problem instance the rtac data is
        populated by the resulting information.

        Returns
        -------
        None
        """
        self.subnow = time.process_time_ns()
        self.rtac_data.ta_rtac_time[self.core] = \
            round(
                (time.process_time_ns() - self.proc_cpu_time) * 10**(-9)
                + self.time, 2)
        self.rtac_data.ev.set()
        self.rtac_data.event = 1
        if not self.scenario.objective_min:
            self.rtac_data.winner.value = self.config_id
        elif self.rtac_data.ta_res[self.core] <= min(self.rtac_data.ta_res[:]):
            self.rtac_data.winner.value = self.config_id
        self.rtac_data.newtime = self.rtac_data.ta_res_time[self.core]
        self.running = False
        self.rtac_data.status[self.core] = 2  # TARunStatus.finished

    def kill_run(self) -> None:
        """
        Terminates this process/ target algorithm run, as well as the other
        contenders. Several layers of termination are included to ensure
        termination on different platforms.

        Returns
        -------
        None
        """
        if self.rtac_data.status[self.core] != 2:
            self.rtac_data.status[self.core] = 3  # TARunStatus.capped
        self.running = False
        self.proc.terminate()
        time.sleep(0.1)
        if self.proc.poll() is None:
            self.proc.kill()
            time.sleep(0.1)
            if not self.scenario.objective_min:
                for ii in range(self.scenario.number_cores):
                    if (self.rtac_data.substart[ii]
                        - time.process_time_ns()) * 1e-9 >= \
                            self.rtac_data.newtime.value and \
                            ii != self.core:
                        os.kill(self.rtac_data.pids[ii], signal.SIGKILL)
            time.sleep(0.1)
            try:
                os.kill(self.pid, signal.SIGKILL)
            except Exception as e:
                self.logs.general_log(e)
                pass

    def run(self, instance: str, config: Configuration,
            rtac_data: RTACData | RTACDatapp, sync_event: Event) -> None:
        """
        Manages the target algorithm runner functions depending on the state
        of the run.

        Parameters
        ----------
        instance : str
            Path to the problem instance to solve.
        config : Configuration
            Configuration of parameter values to run the problem instance with.
        rtac_data : RTACData | RTACDatapp
            Object containing data and objects necessary throughout the RTAC 
            modules.

        Returns
        -------
        None
        """
        sync_event.wait()
        self.start_run(instance, config, rtac_data)
        while self.running:
            # Avoid checking output excessively often (causes too much
            # overhead)
            time.sleep(0.005)  # time.sleep(5e-6)

            ta_output = non_block_read(self.proc.stdout, self.logs)

            self.check_output(ta_output)
            # If result entry is different to default, populate result lists
            if self.rtac_data.ta_res[self.core] != self.max_float:
                self.check_result()
            # If objective minimization scenario
            if self.scenario.objective_min:
                # and if time limit is reached, kill this target algorithm run
                # + 1 sec extra time for the TA to shut down, wrap up and print
                # the result, since it is important to know it
                if time.time() - self.om_start >= self.scenario.timeout + 1:
                    self.rtac_data.status[self.core] = 5  # TARunStatus.timeout
                    self.kill_run()
            # If runtime minimization and one TA run solved instance, kill all
            # target algorithm runs
            elif self.rtac_data.event == 1 or self.rtac_data.ev.is_set():
                time.sleep(2)
                self.kill_run()      


class TARunnerpp(BaseTARunner):
    """
    Target algorithm runner for ReACTR++ implementation.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    logs : RTACLogs
        Object containing loggers and logging functions.
    core : int
        Identifier for the parallel run on a specific core.
    """

    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs,
                 core: int):
        BaseTARunner.__init__(self, scenario, logs, core)
        self.interim_check_increment = scenario.timeout / 150
        self.interim_check_time = time.time()

    def check_output(self, ta_output: bytes) -> None:
        """
        Checks the output, if there was any, and either declares the instance 
        as solved by the contender (if the corresponding marker is present) or 
        outputs intermediate target algorithm output.

        Parameters
        ----------
        ta_output : bytes
            Output from `subprocess.PIPE`.

        Returns
        -------
        None
        """
        if ta_output != b'':
            if_solved = \
                self.wrapper.check_if_solved(ta_output, non_block_read,
                                             self.proc)

            if if_solved is not None:
                res, self.time, self.rtac_data.event = if_solved
                self.rtac_data.ta_res[self.core] = res
                self.rtac_data.ta_res_time[self.core] = self.time
                self.rtac_data.status[self.core] = 2  # TARunStatus.finished

            if time.time() - self.interim_check_time \
                    >= self.interim_check_increment:  # reduce frequency

                self.interim_check_time = time.time()

                interim = self.wrapper.check_output(ta_output)

                if interim is not None:
                    self.rtac_data.interim[self.core] = interim

            if self.scenario.gray_box:
                self.gb_record(ta_output)


def gb_record(self, ta_output: bytes) -> None:
    """
    Records runtime output of the target algorithm if there was any new.

    Parameters
    ----------
    ta_output : bytes
        Output from `subprocess.PIPE`.

    Returns
    ------
    None
    """
    now = time.time()
    elapsed_time = now - self.last_check
    if elapsed_time >= self.scenario.gb_read_time:
        rt_feats = self.wrapper.check_output(ta_output)
        if rt_feats is not None and \
                rt_feats != self.rtac_data.RuntimeFeatures[self.core]:

            self.rtac_data.RuntimeFeatures[self.core] = rt_feats
            CPUTimeExpended = \
                (
                    time.process_time_ns() - self.rtac_data.substart[self.core]
                ) * 1e-9
            ClockTimeExpended = time.time() - self.clock_start

            rt_record = {'core': self.core,
                         'CPUTimeExpended': CPUTimeExpended,
                         'ClockTimeExpended': ClockTimeExpended,
                         'rt_feats': rt_feats}

            if rt_record not in self.rtac_data.rec_data[self.core].values():
                self.rtac_data.rec_data[self.core][int(ClockTimeExpended)] = \
                    rt_record

        self.last_check = now


def ta_runner_factory(scenario: argparse.Namespace, logs: RTACLogs,
                      core: int) -> BaseTARunner | TARunnerpp:
    """
    Class factory to return the initialized target algorithm runner class
    appropriate to the RTAC method specified in `scenario.ac`.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    logs : RTACLogs
        Object containing loggers and logging functions.
    core : int
        Number of the parallel run started on a core.

    Returns
    -------
    BaseTARunner or TARunnerpp
        Initialized target algorithm runner object matching the RTAC method
        of the scenario.
    """
    if scenario.ac in (ACMethod.ReACTR, ACMethod.CPPL):
        tarunner = BaseTARunner
    elif scenario.ac == ACMethod.ReACTRpp:
        tarunner = TARunnerpp

    if scenario.gray_box:
        tarunner.gb_record = gb_record

    return tarunner(scenario, logs, core)


if __name__ == "__main__":
    pass
