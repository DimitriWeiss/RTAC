"""This module implements the target algorithm wrapper for CaDiCaL 1.2.1.""" 

from subprocess import Popen, PIPE
import subprocess
from typing import Any
import time
import sys
import os
from wrapper.abstract_wrapper import AbstractWrapper
from ac_functionalities.rtac_data import Configuration, InterimMeaning
from ac_functionalities.ta_runner import non_block_read

sys.path.append(os.getcwd())


class TSP_RT(AbstractWrapper):
    """Python-TSP Wrapper for runtime minimization scenario. Annealing factor
    'a' is fixed to have a fair comparison of runtime performance."""

    def translate_config(self, config: Configuration) -> list[str]:
        """Convert dictionary representation of the configuration to a list of
        parameter name and value alternating.

        :param config: Configuration object - parameter values to run problem
            instance with.
        :type config: Configuration
        :returns: list of strings representation of the configuration
        :rtype: list[str]
        """
        config_list = []
        config.conf['-a'] = 0.9  # runtime scenario: fixed annealing factor
        for name, param in config.conf.items():
            config_list.append(name)
            config_list.append(str(param))

        return config_list

    def start(self, config: Any, timeout: int,
              instance: str) -> tuple[subprocess.Popen, int]:
        """Start CaDiCaL via subprocess.Popen with stdout to
        subprocess.PIPE, configuration <config> on instance <instance> with
            timelimit <timeout>.

        :param config: Parameters in a format as needed for target algorithm.
        :type config: Any
        :param timeout: Maximum runtime allowed for target algorithm run in
            seconds.
        :type timeout: int
        :param instance: Path to problem instance.
        :type instance: str
        :returns: Target algorithm via subprocess.Popen process and starting
        time of the process
        :rtype: tuple[subprocess.Popen, int]
        """
        proc = Popen(['python3', f'{self.path}/data/solvers/python-tsp.py',
                      *config, '-t', str(timeout), '-i',
                      f'{self.path}/{instance}'],
                     stdout=PIPE)

        self.timeout = timeout

        proc_cpu_time = time.process_time_ns()

        return proc, proc_cpu_time

    def check_if_solved(self, ta_output: bytes, nnr: non_block_read,
                        proc: subprocess.Popen) -> tuple[
                            int | float, float, int] | None:
        """Bytes output of the subprocess.Popen process running CaDiCaL is
        checked to determine if the problem instance is solved (res).

        :param ta_output: Output of the target algorithm.
        :type ta_output: bytes
        :param nnr: Non blocking read function for accessing the
            subprocess.PIPE output of the target algorithm
        :type nnr: non_nlock_read
        :param proc: Target algorithm run via subprocess.Popen process
        :type proc: subprocess.Popen
        :returns: Target algorithm result (1 if solved, 0 else), runtime
            needed and event (0 or 1, if solved), or None
        :rtype: tuple[int | float, float, int]
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
    """Python-TSP Wrapper for cost minimization scenario. Annealing factor
    is not fixed. If TA is much faster than the time limit but still yields a
    better solution it is not a problem."""

    def translate_config(self, config: Configuration) -> list[str]:
        """Overriding TSP_RT function: Convert dictionary representation of
        the configuration to a list of parameter name and value alternating.

        :param config: Configuration object - parameter values to run problem
            instance with.
        :type config: Configuration
        :returns: list of strings representation of the configuration
        :rtype: list[str]
        """
        config_list = []
        for name, param in config.conf.items():
            config_list.append(name)
            config_list.append(str(param))

        return config_list


class TSP_RTpp(TSP_RT):
    """Python-TSP Wrapper for runtime minimization scenario. Annealing factor
    'a' is fixed to have a fair comparison of runtime performance. Additional
    functions for ReACTR++ implementation."""

    def interim_info(self) -> list[InterimMeaning]:
        """Gives information about whether a higher or a lower level of the
        entry is a sign of higher quality of the configuration regarding the
        target algorithm run.

        :return: List of InterimMeaning - is a higher or lower value better.
        :rtype: list[InterimMeaning] or None
        """

        self.interim_meaning = [InterimMeaning.decrease]

        return self.interim_meaning

    def check_output(self, ta_output: bytes) -> list[float] | None:
        """Parsing runtime output of the target algorithm.

        :param ta_output: Output of the target algorithm.
        :type ta_output: bytes
        :return: List of intermediate output values if provided by TA.
        :rtype: list[float] or None
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
    """Python-TSP Wrapper for cost minimization scenario. Annealing factor
    is not fixed. If TA is much faster than the time limit but still yields a
    better solution it is not a problem. Additional functions for ReACTR++
    implementation."""

    def interim_info(self) -> list[InterimMeaning]:
        """Gives information about whether a higher or a lower level of the
        entry is a sign of higher quality of the configuration regarding the
        target algorithm run.

        :return: List of InterimMeaning - is a higher or lower value better.
        :rtype: list[InterimMeaning] or None
        """
        self.interim_meaning = [InterimMeaning.decrease,
                                InterimMeaning.increase,
                                InterimMeaning.decrease,
                                InterimMeaning.increase]

        return self.interim_meaning

    def check_output(self, ta_output) -> list[float] | None:
        """Parsing runtime output of the target algorithm.

        :param ta_output: Output of the target algorithm.
        :type ta_output: bytes
        :return: List of intermediate output values if provided by TA.
        :rtype: list[float] or None
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
