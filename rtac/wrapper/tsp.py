"""This module implements the target algorithm wrapper for CaDiCaL 1.2.1.""" 

from subprocess import Popen, PIPE
import subprocess
from typing import Any
import time
import sys
import os
from rtac.wrapper.abstract_wrapper import AbstractWrapper
from rtac.ac_functionalities.rtac_data import Configuration, InterimMeaning
from rtac.ac_functionalities.ta_runner import non_block_read

sys.path.append(os.getcwd())


class TSP_RT(AbstractWrapper):
    """
    Python-TSP Wrapper for runtime minimization scenario. Annealing factor
    'a' is fixed to have a fair comparison of runtime performance.
    """

    def translate_config(self, config: Configuration) -> list[str]:
        """
        Convert dictionary representation of the configuration to a list of
        parameter names and values alternating.

        Parameters
        ----------
        config : Configuration
            Configuration object - parameter values to run problem instance 
            with.

        Returns
        -------
        list of str
            List of strings representation of the configuration.
        """
        config_list = []
        config.conf['-a'] = 0.9  # runtime scenario: fixed annealing factor
        for name, param in config.conf.items():
            config_list.append(name)
            config_list.append(str(param))

        return config_list

    def start(self, config: Any, timeout: int,
              instance: str) -> tuple[subprocess.Popen, int]:
        """
        Start CaDiCaL via subprocess.Popen with stdout set to subprocess.PIPE,
        using the given configuration on the specified instance with a time 
        limit.

        Parameters
        ----------
        config : Any
            Parameters in the format required by the target algorithm.
        timeout : int
            Maximum runtime allowed for the target algorithm run in seconds.
        instance : str
            Path to the problem instance.

        Returns
        -------
        tuple of (subprocess.Popen, int)
            Target algorithm subprocess.Popen process and the start time of 
            the process.
        """
        # Absolute path to the current file
        file_path = os.path.abspath(__file__)

        # Directory containing the file
        file_dir = os.path.dirname(file_path)
        file_dir = file_dir.split('wrapper')[0]

        proc = Popen(['python3',
                      f'{file_dir}data/solvers/python-tsp.py',
                      *config, '-t', str(timeout), '-i',
                      f'{file_dir}{instance}'],
                     stdout=PIPE)

        self.timeout = timeout

        proc_cpu_time = time.process_time_ns()

        return proc, proc_cpu_time

    def check_if_solved(self, ta_output: bytes, nnr: non_block_read,
                        proc: subprocess.Popen) -> tuple[
                            int | float, float, int] | None:
        """
        Check the bytes output of the subprocess.Popen process running CaDiCaL
        to determine if the problem instance is solved.

        Parameters
        ----------
        ta_output : bytes
            Output of the target algorithm.
        nnr : non_nlock_read
            Non-blocking read function for accessing the subprocess.PIPE 
            output.
        proc : subprocess.Popen
            Target algorithm run via subprocess.Popen process.

        Returns
        -------
        tuple of (int or float, float, int) or None
            Target algorithm result (1 if solved, 0 otherwise), runtime needed,
            and event (0 or 1, indicating if solved), or None if no result.
        """
        if ta_output != b'':
            b = str(ta_output.strip())
            if 'Warning' in b:  # Appears in b, if TA reaches time limit
                time = self.timeout
                res = sys.maxsize
                event = 0

                return res, time, event

            if 'Time:' in b:
                time = float(b.split(' ')[1][:-1])
                res_not_given = True
                while res_not_given:
                    line = nnr(proc.stdout)
                    b = str(line.strip())
                    if 'Distance:' in b:
                        res = float(b.split(' ')[1][:-1])
                        res_not_given = False

                event = 1
                proc.stdout.close()

            else:

                return None

            return res, time, event
        else:
            return None


class TSP_Q(TSP_RT):
    """
    Python-TSP Wrapper for cost minimization scenario. Annealing factor
    is not fixed. If TA is much faster than the time limit but still yields a
    better solution it is not a problem.
    """

    def translate_config(self, config: Configuration) -> list[str]:
        """
        Overriding TSP_RT function: Convert dictionary representation of
        the configuration to a list of parameter name and value alternating.

        Parameters
        ----------
        config : Configuration
            Configuration object – parameter values to run the problem 
            instance with.

        Returns
        -------
        list of str
            List representation of the configuration with parameter names and 
            values alternating.
        """
        config_list = []
        for name, param in config.conf.items():
            config_list.append(name)
            config_list.append(str(param))

        return config_list


class TSP_RTpp(TSP_RT):
    """
    Python-TSP Wrapper for runtime minimization scenario. Annealing factor
    'a' is fixed to have a fair comparison of runtime performance. Additional
    functions for ReACTR++ implementation.
    """

    def interim_info(self) -> list[InterimMeaning]:
        """
        Gives information about whether a higher or a lower level of the
        entry is a sign of higher quality of the configuration regarding the
        target algorithm run.

        Returns
        -------
        list of InterimMeaning or None
            Indicates if a higher or lower value is better.
        """
        self.interim_meaning = [InterimMeaning.decrease]

        return self.interim_meaning

    def check_output(self, ta_output: bytes) -> list[float] | None:
        """
        Parse runtime output of the target algorithm.

        Parameters
        ----------
        ta_output : bytes
            Output of the target algorithm.

        Returns
        -------
        list of float or None
            List of intermediate output values if provided by the target 
            algorithm.
        """
        if ta_output != b'':
            b = str(ta_output.strip())
            # Check for progress
            if 'Temperature' in b:
                b = b.split(' ')
                # Assumption: the lower the temperature, the closer the TA is
                # to finding the solution. Solution Quality is not regarded in
                # this example, we optimize for runtime.
                temp = float(b[1][:-1])
                interim = [temp]

                return interim
            else:
                return None
        else:
            return None


class TSP_Qpp(TSP_Q):
    """
    Python-TSP Wrapper for cost minimization scenario. Annealing factor
    is not fixed. If TA is much faster than the time limit but still yields a
    better solution it is not a problem. Additional functions for ReACTR++
    implementation.
    """

    def interim_info(self) -> list[InterimMeaning]:
        """
        Gives information about whether a higher or a lower level of the
        entry is a sign of higher quality of the configuration regarding the
        target algorithm run.

        Returns
        -------
        list of InterimMeaning or None
            Indicates if a higher or lower value is better.
        """
        self.interim_meaning = [InterimMeaning.decrease,
                                InterimMeaning.increase,
                                InterimMeaning.decrease,
                                InterimMeaning.increase]

        return self.interim_meaning

    def check_output(self, ta_output) -> list[float] | None:
        """
        Parse runtime output of the target algorithm.

        Parameters
        ----------
        ta_output : bytes
            Output of the target algorithm.

        Returns
        -------
        list of float or None
            List of intermediate output values if provided by the target 
            algorithm.
        """
        if ta_output != b'':
            b = str(ta_output.strip())
            # Check for progress
            if 'Temperature' in b:
                b = b.split(' ')
                
                temp = float(b[1][:-1])
                k = float(b[6].split('/')[0])
                k_acc = float(b[8].split('/')[0])
                k_noimp = float(b[10][:-1])
                interim = [temp, k, k_acc, k_noimp]

                return interim
            else:
                return None
        else:
            return None


if __name__ == '__main__':
    pass
