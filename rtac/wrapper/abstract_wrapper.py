"""The abstract target algorithm runner class is defined in this module."""

from abc import ABC, abstractmethod
from typing import Any
from subprocess import Popen, PIPE
import subprocess
import time
import sys
import os
from rtac.ac_functionalities.ta_runner import non_block_read
from rtac.ac_functionalities.rtac_data import Configuration


class AbstractWrapper(ABC):
    """Abstract target algorithm wrapper class."""

    def __init__(self):
        """Make sure target algorithm is executble by using absolute path to
        target algorithm."""
        sys.path.append(os.getcwd())
        self.path = sys.path[-1]

    @abstractmethod
    def translate_config(self, config: Configuration) -> Any:
        """Convert dictionary representation of the configuration to the format
        needed by the wrapper to pass to the target algorithm.

        :param config: Configuration object of parameter values to run
            problem instance with.v
        :type config: Configuration
        """

    @abstractmethod
    def start(self, params: Any, timelimit: int,
              instance: str) -> tuple[subprocess.Popen, int]:
        """Start the target algorithm via subprocess.Popen with stdout to
        subprocess.PIPE.

        :param params: Parameters in a format as needed for target algorithm.
        :type params: Any
        :param timelimit: Maximum runtime allowed for target algorithm run in
            seconds.
        :type timelimit: int
        :param instance: Path to problem instance.
        :type instance: str
        :returns: Target algorithm via subprocess.Popen process and starting
        time of the process
        :rtype: tuple[subprocess.Popen, int]
        """
        proc = Popen(['echo', 'Hello World!'],
                     stdout=PIPE)

        proc_cpu_time = time.process_time()

        return proc, proc_cpu_time

    @abstractmethod
    def check_if_solved(self, ta_output: bytes, nnr: non_block_read,
                        proc: subprocess.Popen) -> tuple[
                            int | float, float, int] | None:
        """Bytes output of the subprocess.Popen process running the target
        algorithm is checked to determine if the problem instance is solved.

        :param ta_output: Output of the target algorithm.
        :type ta_output: bytes
        :param nnr: Non blocking read function for accessing the
            subprocess.PIPE output of the target algorithm
        :type nnr: non_nlock_read
        :param proc: Target algorithm run via subprocess.Popen process
        :type proc: subprocess.Popen
        :returns: Target algorithm result, runtime needed and event
            (0 or 1, if solved), or None
        :rtype: tuple[int | float, float, int]
        """
        if ta_output != b'':  # Check if output is not empty bytes
            result = 0
            time = 0.0
            event = 0

            return result, time, event
        else:
            return None


'''
class AbstractWrapperpp(AbstractWrapper):

    @abstractmethod
    def check_output(ta_output):
        """Parsing runtime output of the solver."""
        if ta_output != b'':
            interim = []

            return interim
        else:
            return 'No output'
'''
