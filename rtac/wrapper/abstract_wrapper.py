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
    """
    Abstract target algorithm wrapper class.
    """

    def __init__(self):
        """
        Make sure target algorithm is executble by using absolute path to
        target algorithm.
        """
        sys.path.append(os.getcwd())
        self.path = sys.path[-1]

    @abstractmethod
    def translate_config(self, config: Configuration) -> Any:
        """
        Convert dictionary representation of the configuration to the format
        needed by the wrapper to pass to the target algorithm.

        Parameters
        ----------
        config : Configuration
            Configuration object of parameter values to run problem instance 
            with.

        Returns
        -------
        Any
            Any form of the configuration that is needed by the target 
            algorithm.
        """

    @abstractmethod
    def start(self, params: Any, timelimit: int,
              instance: str) -> tuple[subprocess.Popen, int]:
        """
        Start the target algorithm via subprocess.Popen with stdout to
        subprocess.PIPE.

        Parameters
        ----------
        params : Any
            Parameters in a format as needed for target algorithm.
        timelimit : int
            Maximum runtime allowed for target algorithm run in seconds.
        instance : str
            Path to problem instance.

        Returns
        -------
        tuple
            - **proc** : subbrocess.Process,
              The process started with the target algorithm
            - **proc_cpu_time** : int,
              CPU time of the subprocess.
        """
        proc = Popen(['echo', 'Hello World!'],
                     stdout=PIPE)

        proc_cpu_time = time.process_time()

        return proc, proc_cpu_time

    @abstractmethod
    def check_if_solved(self, ta_output: bytes, nnr: non_block_read,
                        proc: subprocess.Popen) -> tuple[
                            int | float, float, int] | None:
        """
        Bytes output of the subprocess.Popen process running the target
        algorithm is checked to determine if the problem instance is solved.

        Parameters
        ----------
        ta_output : bytes
            Output of the target algorithm.
        nnr : non_nlock_read
            Non-blocking read function for accessing the subprocess.PIPE output
            of the target algorithm.
        proc : subprocess.Popen
            Target algorithm run via subprocess.Popen process.

        Returns
        -------
        tuple or None
            - **result** : int | float,
              Objective value.
            - **time** : float,
              Runtime needed.
            - **event** : int,
              0 or 1, if solved.
        """
        if ta_output != b'':  # Check if output is not empty bytes
            result = 0
            time = 0.0
            event = 0

            return result, time, event
        else:
            return None
