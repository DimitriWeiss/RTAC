"""In this module classes to run the target algorithm and observe its output
through the wrapper are implemented."""

from abc import ABC, abstractmethod
from typing import Any
import fcntl
import os
import sys
import time
import signal
import importlib
import argparse
from ac_functionalities.rtac_data import ACMethod, Configuration, RTACData
from ac_functionalities.logs import RTACLogs


def non_block_read(ta_output: bytes, logs: RTACLogs = None) -> str:
    """Function for reading subprocess.PIPE output without blocking the
    application until there is output. It looks up if there is output and
    disconnects.

    :param ta_output: subprocess.PIPE output.
    :type ta_output: bytes
    :param logs: Object containing loggers and logging functions.
    :type: RTACLogs
    :returns: Either output as string or an empty string if nothing was there.
    :rytpe: str
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
    """Abstract TARunner."""

    @abstractmethod
    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs,
                 core: int) -> None:
        """Initializes target algorithm runner, loads and instantiates target
        algorithm wrapper.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        :param logs: Object containing loggers and logging functions.
        :type: RTACLogs
        :param core: Number of the parallel run started on a core.
        :type: int
        """

    @abstractmethod
    def translate_config(self, config: Configuration) -> Any:
        """Convert dictionary representation of the configuration to the format
        needed by the wrapper to pass to the target algorithm.

        :param config: Configuration of parameter values to run problem
            instance with.
        :type config: Configuration
        :return: new representation of the configuration
        :rtype: Any
        """

    @abstractmethod
    def start_run(self, instance: str, config: Configuration,
                  rtac_data: RTACData) -> None:
        """Starts target algorithm run and populates data necessary for
        corination of tournament members.

        :param instance: path to the problem instance to solve.
        :type instance: str
        :param config: Representation of the configuration
        :type: Any
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        """

    @abstractmethod
    def check_output(self, ta_output: bytes) -> None:
        """Checks the output, if there was any, and declares instance as solved
        by the contender, if the marker for it was present.

        :param ta_output: subprocess.PIPE output.
        :type ta_output: bytes
        """

    @abstractmethod
    def check_result(self) -> None:
        """If this contender solved the problem instance the rtac data is
        populated by the resulting information according to the RTAC method
        scenario.ac."""

    @abstractmethod
    def kill_run(self) -> None:
        """Terminates this process/ target algorithm run, as well as the other
        contenders. Several layers of termination are included to ensure
        termination on different platforms."""

    @abstractmethod
    def run(self, instance: str, config: Configuration,
            rtac_data: RTACData) -> None:
        """Manages the target algorithm runner functions depending on state of
        the run according to the RTAC method scenario.ac.

        :param instance: path to the problem instance to solve.
        :type instance: str
        :param config: Configuration of parameter values to run problem
            instance with.
        :type config: Configuration
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        """


class BaseTARunner(AbstractTARunner):
    """Target algorithm runner for ReACTR implementation."""
    
    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs,
                 core: int) -> None:
        """Initializes target algorithm runner, loads and instantiates target
        algorithm wrapper.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        :param logs: Object containing loggers and logging functions.
        :type: RTACLogs
        :param core: Number of the parallel run started on a core.
        :type: int
        """
        self.scenario = scenario
        self.logs = logs
        self.core = core
        self.max_float = sys.float_info.max * 1e-100
        module = importlib.import_module(scenario.wrapper)
        name = scenario.wrapper_name
        self.wrapper = getattr(module, name)()

    def translate_config(self, config: Configuration) -> Any:
        """Convert dictionary representation of the configuration to the format
        needed by the wrapper to pass to the target algorithm.

        :param config: Configuration of parameter values to run problem
            instance with.
        :type config: Configuration
        :return: new representation of the configuration
        :rtype: Any
        """
        self.config_id = config.id
        translated_config = self.wrapper.translate_config(config)

        return translated_config

    def start_run(self, instance: str, config: Any,
                  rtac_data: RTACData) -> None:
        """Starts target algorithm run and populates data necessary for
        corination of tournament members.

        :param instance: path to the problem instance to solve.
        :type instance: str
        :param config: Representation of the configuration
        :type: Any
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        """
        self.config = config
        self.instance = instance
        self.rtac_data = rtac_data

        self.rtac_data.substart[self.core] = time.process_time_ns()
        if self.scenario.objective_min:
            self.om_start = time.time()

        self.proc, self.proc_cpu_time = \
            self.wrapper.start(self.config, self.scenario.timeout,
                               self.instance)
        
        self.pid = self.proc.pid
        self.rtac_data.pids[self.core] = self.pid
        self.running = True
        self.rtac_data.status[self.core] = 1

    def check_output(self, ta_output: bytes) -> None:
        """Checks the output, if there was any, and declares instance as solved
        by the contender, if the marker for it was present.

        :param ta_output: subprocess.PIPE output.
        :type ta_output: bytes
        """
        if ta_output != b'':
            if_solved = \
                self.wrapper.check_if_solved(ta_output, non_block_read,
                                             self.proc)

            if if_solved is not None:
                res, self.time, self.rtac_data.event = if_solved
                self.rtac_data.ta_res[self.core] = res
                self.rtac_data.ta_res_time[self.core] = self.time
                self.rtac_data.status[self.core] = 2

    def check_result(self) -> None:
        """If this contender solved the problem instance the rtac data is
        populated by the resulting information."""
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
        self.rtac_data.status[self.core] = 2

    def kill_run(self) -> None:
        """Terminates this process/ target algorithm run, as well as the other
        contenders. Several layers of termination are included to ensure
        termination on different platforms."""
        if self.rtac_data.status[self.core] != 2:
            self.rtac_data.status[self.core] = 3
        self.running = False
        self.proc.terminate()
        time.sleep(0.1)
        if self.proc.poll() is None:
            self.proc.kill()
            time.sleep(0.1)
            if not self.scenario.objective_min:
                for ii in range(self.scenario.number_cores):
                    if self.rtac_data.substart[ii] - time.time() >= \
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
            rtac_data: RTACData) -> None:
        """Manages the target algorithm runner functions depending on state of
        the run.

        :param instance: path to the problem instance to solve.
        :type instance: str
        :param config: Configuration of parameter values to run problem
            instance with.
        :type config: Configuration
        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        """
        self.start_run(instance, config, rtac_data)
        while self.running:
            # Avoid checking output excessively often (cause  too much
            # overhead)
            time.sleep(5e-6)  

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
                    self.kill_run()
            # If runtime minimization and one TA run solved instance, kill all
            # target algorithm runs
            elif self.rtac_data.event == 1 or self.rtac_data.ev.is_set():
                self.kill_run()      


class TARunnerpp(BaseTARunner):
    """Target algorithm runner for ReACTR++ implementation."""

    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs,
                 core: int) -> None:
        BaseTARunner.__init__(self, scenario, logs, core)
        self.interim_check_increment = scenario.timeout / 150
        self.interim_check_time = time.time()

    def check_output(self, ta_output: bytes) -> None:
        """Checks the output, if there was any, and declares instance as solved
        by the contender, if the marker for it was present or outputs
        intermediate target algorithm output.

        :param ta_output: subprocess.PIPE output.
        :type ta_output: bytes
        """
        if ta_output != b'':
            if_solved = \
                self.wrapper.check_if_solved(ta_output, non_block_read,
                                             self.proc)

            if if_solved is not None:
                res, self.time, self.rtac_data.event = if_solved
                self.rtac_data.ta_res[self.core] = res
                self.rtac_data.ta_res_time[self.core] = self.time
                self.rtac_data.status[self.core] = 2

            if time.time() - self.interim_check_time \
                    >= self.interim_check_increment:  # reduce frequency

                self.interim_check_time = time.time()

                interim = self.wrapper.check_output(ta_output)

                if interim is not None:
                    self.rtac_data.interim[self.core] = interim


'''
class GBTARunner(TARunnerpp):
    """Target algorithm runner for GrayBox implementation."""

    def check_output(self, scenario):
        self.line = non_block_read(self.proc.stdout)

    def kill_run(self, scenario):
        self.proc.kill()

    def run(self, scenario):
        self.proc = self.start_run()
        while True:

            self.check_output()
            self.check_result()
            if self.a == 1:
                self.kill_run()
'''


def ta_runner_factory(scenario: argparse.Namespace, logs: RTACLogs,
                      core: int) -> AbstractTARunner:
    """Class factory to return the initialized target algorithm runner class
    appropriate to the RTAC method scenario.ac.

    :param scenario: Namespace containing all settings for the RTAC.
    :type scenario: argparse.Namespace
    :param logs: Object containing loggers and logging functions.
    :type: RTACLogs
    :param core: Number of the parallel run started on a core.
    :type: int
    :returns: Inititialized AbstractTARunner object matching the RTAC method of
        the scenario.
    :rtype: AbstractTARunner
    """
    if scenario.ac in (ACMethod.ReACTR, ACMethod.CPPL):
        return BaseTARunner(scenario, logs, core)

    elif scenario.ac == ACMethod.ReACTRpp:
        return TARunnerpp(scenario, logs, core)

    '''

    elif scenario.ac == ACMethod.GRAYBOX:
        return GBTARunner(scenario)
    '''


if __name__ == "__main__":
    pass
