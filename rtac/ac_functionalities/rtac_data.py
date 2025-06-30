"""This module contains data structures needed for the RTAC."""

from abc import ABC, abstractmethod
import time
import copy
from typing import Optional
from enum import Enum
from dataclasses import dataclass, field
from multiprocessing import (
    freeze_support,
    Event, Manager,
    Array, Value
)
from uuid import UUID
import argparse
import sys


class ACMethod(Enum):
    """
    Enumeration for the RAC methods.

    Members
    -------
    ReACTR : int
        Represents the 'ReACTR' option.
    ReACTRpp : int
        Represents the 'ReACTR++' option.
    CPPL : int
        Represents the 'CPPL' option.
    """
    ReACTR = 1
    ReACTRpp = 2
    CPPL = 3


class Distribution(Enum):
    """
    Enumeration for the distribution types to draw random values from.

    Members
    -------
    uniform : int
        Represents the uniform distribution option.
    log : int
        Represents the logarithmic distribution option.
    """
    uniform = 1
    log = 2


class ValType(Enum):
    """
    Enumeration for the type of values randomly generated.

    Members
    -------
    str : int
        Represents the string option.
    int : int
        Represents the integer option.
    """
    str = 1
    int = 2


class ParamType(Enum):
    """
    Enumeration for the parameter type to be generated.

    Members
    -------
    discrete : int
        Represents the string option.
    continuous : int
        Represents the integer option.
    categorical : int
        Represents the string option.
    binary : int
        Represents the integer option.
    """
    discrete = 1
    continuous = 2
    categorical = 3
    binary = 4


@dataclass
class DiscreteParameter:
    """
    Represents a discrete (integer) parameter with optional default,
    sampling behavior, and distributional assumptions.

    Parameters
    ----------
    paramtype : str
        A string identifying the type of parameter.
    minval : int
        Minimum allowed value for the parameter.
    maxval : int
        Maximum allowed value for the parameter.
    default : int, optional
        Default value used, if specified.
    splitbydefault : bool, optional
        If True, use the default to split the parameter range for logarithmic 
        distribution draws.
    distribution : Distribution, optional
        Distribution used for sampling (e.g., uniform, triangular). Defaults 
        to uniform.
    logonpos : bool, optional
        If True, apply logarithmic sampling on the positive part of the 
        parameter range.
    logonneg : bool, optional
        If True, apply logarithmic sampling on the negative part of the 
        parameter range.
    probabpos : float, optional
        Probability of sampling from positive part of the range.
    probabneg : float, optional
        Probability of sampling from negative part of the range.
    includezero : bool, optional
        If True, include zero as a possible sampled value.
    probabilityzero : float, optional
        Probability of sampling zero if `includezero` is True.
    """
    paramtype: str
    minval: int
    maxval: int
    default: Optional[int] = None
    splitbydefault: Optional[bool] = False
    distribution: Optional[Distribution] = Distribution.uniform
    logonpos: Optional[bool] = False
    logonneg: Optional[bool] = False
    probabpos: Optional[float] = 0.49
    probabneg: Optional[float] = 0.49
    includezero: Optional[bool] = False
    probabilityzero: Optional[float] = 0.02


@dataclass
class ContinuousParameter:
    """
    Represents a continuous (real) parameter with optional default,
    sampling behavior, and distributional assumptions.

    Parameters
    ----------
    paramtype : str
        A string identifying the type of parameter.
    minval : int
        Minimum allowed value for the parameter.
    maxval : int
        Maximum allowed value for the parameter.
    default : int, optional
        Default value used, if specified.
    splitbydefault : bool, optional
        If True, use the default to split the parameter range for logarithmic 
        distribution draws.
    distribution : Distribution, optional
        Distribution used for sampling (e.g., uniform, triangular). Defaults 
        to uniform.
    logonpos : bool, optional
        If True, apply logarithmic sampling on the positive part of the 
        parameter range.
    logonneg : bool, optional
        If True, apply logarithmic sampling on the negative part of the 
        parameter range.
    probabpos : float, optional
        Probability of sampling from positive part of the range.
    probabneg : float, optional
        Probability of sampling from negative part of the range.
    includezero : bool, optional
        If True, include zero as a possible sampled value.
    probabilityzero : float, optional
        Probability of sampling zero if `includezero` is True.
    """
    paramtype: str
    minval: float
    maxval: float
    default: Optional[float] = None
    splitbydefault: Optional[bool] = None
    distribution: Optional[Distribution] = Distribution.uniform
    logonpos: Optional[bool] = False
    logonneg: Optional[bool] = False
    probabpos: Optional[float] = 0.49
    probabneg: Optional[float] = 0.49
    includezero: Optional[bool] = False
    probabilityzero: Optional[float] = 0.02


@dataclass
class CategoricalParameter:
    """
    Represents a categorical parameter with optional default.

    Parameters
    ----------
    paramtype : str
        A string identifying the type of parameter.
    flag : bool
        True if parameter is a flag, False if not.
    valtype : ValType
        Identifies the value type.
    default : str | int, optional
        Default value used, if specified.
    minval : int, optional
        Minimum allowed value for the parameter.
    maxval : int, optional
        Maximum allowed value for the parameter.
    values : list[str | int]
        List of the possible values the parameter can assume.
    """
    paramtype: str
    flag: bool = False
    valtype: ValType = ValType.str
    default: Optional[str | int] = None
    minval: Optional[int] = None
    maxval: Optional[int] = None
    values: list[str | int] = field(default_factory=list)


@dataclass
class BinaryParameter:
    """
    Represents a binary parameter with optional default.

    Parameters
    ----------
    paramtype : str
        A string identifying the type of parameter.
    default : str | int, optional
        Default value used, if specified.
    valtype : ValType
        Identifies the value type.
    values : list[str | int]
        List of the possible values the parameter can assume.
    """
    paramtype: str
    default: Optional[str | int]
    valtype: ValType = ValType.int
    values: list[str | int] = field(default_factory=list)


class Parameter(Enum):
    """
    Enumeration for the parameter type to be generated.

    Members
    -------
    discrete : DiscreteParameter
        Represents an string parameter.
    continuous : ContinuousParameter
        Represents an integer parameter.
    categorical : CategoricalParameter
        Represents an string parameter.
    binary : BinaryParameter
        Represents an integer parameter.
    """
    discrete = DiscreteParameter
    continuous = ContinuousParameter
    categorical = CategoricalParameter
    binary = BinaryParameter


class Generator(Enum):
    """
    Enumeration for configuration generator type.

    Members
    -------
    default : DiscreteParameter
        Represents the default configuration generator.
    random : ContinuousParameter
        Represents the random configuration generator.
    crossover : CategoricalParameter
        Represents the genetic crossover configuration generator.
    cppl : BinaryParameter
        Represents the surrogate-assisted genetic crossover configuration 
        generator in CPPL.
    """
    default = 0
    random = 1
    crossover = 2
    cppl = 3


@dataclass
class Configuration:
    """
    Represents a Configuration.

    Parameters
    ----------
    id : UUID
        Unique identifier.
    conf : dict
        Dictionary with parameter name as key and parameter value as value.
    features : list
        Features associated with the configuration.
    gen : Generator
        Configuration generator type used to generate this configuration.
    gen_tourn : int
        Tournament number in which the configuration was generated.
    """
    id: UUID
    conf: dict
    features: list
    gen: Generator
    gen_tourn: int


class TARunStatus(Enum):
    """
    Enumeration for the state of the configuration run.

    Members
    -------
    running : DiscreteParameter
        Configuration is started and running.
    finished : ContinuousParameter
        Configuration has finished the problem instance.
    capped : CategoricalParameter
        Configuration run has been terminated due to another configuration 
        being finished.
    terminated : BinaryParameter
        Configuration has been terminated by gray box.
    timeout : BinaryParameter
        Configuration has been terminated because the time limit has been 
        reached
    awaiting_start : BinaryParameter
        Configuration has not been started yet in an early starting tournament.
    """
    running = 1
    finished = 2
    capped = 3
    terminated = 4
    timeout = 5
    awaiting_start = 6


@dataclass
class TARun:
    """
    Represents a target algorithm run.

    Parameters
    ----------
    config_id : str
        Unique identifier derived from an UUID.
    config : dict
        Dictionary with parameter name as key and parameter value as value.
    res : int | float
        Tournament result (objective value) of the target algorithm run.
    time : float
        Time when the target algorithm run stopped. Set to time limit if the 
        configuration did not solve the problem instance.
    status : TARunStatus
        State of the configuration.
    """
    config_id: str
    config: dict
    res: int | float
    time: float
    status: TARunStatus


@dataclass
class TournamentStats:
    """
    Represents the stats of a tournament.

    Parameters
    ----------
    id : str
        Unique identifier derived from an UUID.
    tourn_nr : int
        Number of the tournament since RTAC init.
    configs : list[UUID]
        List of configuration IDs in the tournament.
    winner : str
        ID of the tournament winner.
    results : list[int | float]
        Objective values of the contenders.
    times : list[float]
        Runtimes of the contenders.
    rtac_times : list[float]
        Runtimes of the contenders + overhead of the configurator during the 
        tournament.
    kills : list[UUID]
        IDs of the contenders terminated by the gray box.
    TARuns : dict[str: TARun]
        List of TARun objects corresponding to 'configs'.
    """
    id: UUID
    tourn_nr: int
    configs: list[UUID]
    winner: str
    results: list[int | float]
    times: list[float]
    rtac_times: list[float]
    kills: list[UUID]
    TARuns: dict[str: TARun]


class InterimMeaning(Enum):
    """
    Enumeration for the meaning of the interim output: Whether increase or 
    decrease signifies progress. Only used in ReACTR++.

    Members
    -------
    increase : DiscreteParameter
        Represents that increase of the value is progress.
    decrease : ContinuousParameter
        Represents that decrease of the value is progress.
    """
    increase = 1
    decrease = 2


class AbstractRTACData(ABC):
    """
    Abstract class to handle picklable data structures needed to coordinate
    and process tournaments.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    """

    @abstractmethod
    def __init__(self, scenario: argparse.Namespace):
        """Initialize all data structures needed for the RTAC."""


class RTACData(AbstractRTACData):
    """
    Class to handle picklable data structures needed to coordinate
    and process tournaments of the ReACTR implementation.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    **kwargs
            Additional keyword arguments that vary by RTAC method.
    """

    def __init__(self, scenario: argparse.Namespace, **kwargs):
        """
        Initialize all data structures needed for ReACTR tournaments.
        """
        huge_res = sys.float_info.max * 1e-100
        self.ev = Event()
        freeze_support()
        # Using int as flags (event), since ctypes do not allow for
        # enum objects.
        self.tournID = 0
        self.cores_start = Manager().list(
            [core for core in range(scenario.number_cores)])
        self.early_start_tournament = Value('b', False)  # False
        self.event = Value('i', 0)
        self.newtime = Value('d', float(scenario.timeout))
        self.best_res = Value('d', huge_res)
        self.winner = Manager().Value('c', 0)
        self.status = \
            Array('i', [0 for core in range(scenario.number_cores)])
        self.pids = \
            Array('i', [0 for core in range(scenario.number_cores)])
        self.substart = \
            Array('d', [0.0 for core in range(scenario.number_cores)])
        self.substart_wall = \
            Array('d', [0.0 for core in range(scenario.number_cores)])
        self.ta_res = \
            Array('d', [huge_res
                        for core in range(scenario.number_cores)])
        self.ta_res_time = \
            Array('d', [scenario.timeout * scenario.runtimePAR
                        for core in range(scenario.number_cores)])
        self.ta_rtac_time = \
            Array('d', [scenario.timeout * scenario.runtimePAR
                        for core in range(scenario.number_cores)])

        # Initialize parallel solving data
        self.process = ['process_{0}'.format(s) 
                        for s in range(scenario.number_cores)]
        self.start = time.time()
        self.winner_known = True
        self.skip = False


class RTACDatapp(RTACData):
    """
    Class to handle picklable data structures needed to coordinate
    and process tournaments of the ReACTR++ implementation.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    **kwargs
            Additional keyword arguments that vary by RTAC method.
    """

    def __init__(self, scenario: argparse.Namespace, **kwargs) -> None:
        RTACData.__init__(self, scenario)
        """
        Initialize additional data structures needed for ReACTR++
        tournaments.
        """
        self.interim_meaning = kwargs.get('interim_meaning')
        #self.interim_weights = interim_weights
        self.interim = Manager().list(
            [[None for _ in range(len(self.interim_meaning))]
             for core in range(scenario.number_cores)])

        # Initialize parallel solving data
        self.interim_res = [[0 for s in range(3)]
                            for c in range(scenario.number_cores)]


class GBData(RTACData):
    """
    Class to handle picklable data structures needed to coordinate
    and process tournaments of the Gray-Box implementation.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    rtacdata_init : AbstractRTACData.__init__
        __init__ of the parent class.
    **kwargs
            Additional keyword arguments that vary by RTAC method.
    """

    def __init__(self, scenario: argparse.Namespace,
                 rtacdata_init: AbstractRTACData.__init__, **kwargs):
        rtacdata_init(self, scenario, **kwargs)
        """
        Initialize additional data structures needed for Gray-Box
        tournaments.
        """
        self.scenario = scenario
        self.rec_data = {core: Manager().dict()
                         for core in range(scenario.number_cores)}

        self.RuntimeFeatures = Manager().list(
            [[] for core in range(scenario.number_cores)])

    def early_rtac_copy(self):
        """
        Returns a modified copy of self to be used in an early starting 
        tournament after terminations by gray box.

        Returns
        -------
        AbstractRTACData
            A modified copy of itself to be used in an early starting 
            tournament after terminations by gray box.

        """
        early_rtac_data = self.__class__.__new__(self.__class__)

        overrides = {
            'ev': Event(),
            'cores_start': self.cores_start,
            'early_start_tournament': Value('b', False),
            'event': Value('i', 0),
            'newtime': Value('d', float(self.scenario.timeout)),
            'best_res': Value('d', sys.float_info.max * 1e-100),
            'winner': Manager().Value('c', 0),
            'status': Array('i',
                            [0 for core in range(self.scenario.number_cores)]),
            'pids': Array('i',
                          [0 for core in range(self.scenario.number_cores)]),
            'substart':
            Array('d', [0.0 for core in range(self.scenario.number_cores)]),
            'substart_wall':
            Array('d', [0.0 for core in range(self.scenario.number_cores)]),
            'ta_res':
            Array('d', [sys.float_info.max * 1e-100
                        for core in range(self.scenario.number_cores)]),
            'ta_res_time':
            Array('d', [self.scenario.timeout * self.scenario.runtimePAR
                        for core in range(self.scenario.number_cores)]),
            'ta_rtac_time':
            Array('d', [self.scenario.timeout * self.scenario.runtimePAR
                        for core in range(self.scenario.number_cores)]),
            'process':
            ['process_{0}'.format(s)
             for s in range(self.scenario.number_cores)],
            'start': time.time()
        }

        # Selectively deepcopy or shallow copy attributes
        for key, value in self.__dict__.items():
            if key in overrides:
                setattr(early_rtac_data, key, overrides[key])
            else:
                # Safe to deepcopy
                setattr(early_rtac_data, key, copy.deepcopy(value))

        return early_rtac_data


def rtacdata_factory(scenario: argparse.Namespace, **kwargs) \
        -> RTACData | RTACDatapp:
    """
    Class factory to return the initialized class with data structures
    appropriate to the RTAC method `scenario.ac`.

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    **kwargs
            Additional keyword arguments that vary by RTAC method.

    Returns
    -------
    RTACData or RTACDatapp
        Initialized AbstractRTACData object matching the RTAC method
        of the scenario.
    """

    if scenario.ac in (ACMethod.ReACTR, ACMethod.CPPL):
        rtacdata = copy.deepcopy(RTACData)
    elif scenario.ac == ACMethod.ReACTRpp:
        rtacdata = copy.deepcopy(RTACDatapp)

    if scenario.gray_box:

        class rtacdata_copy(rtacdata):
            """Copy of the rtacdata class."""

        rtacdata_init = rtacdata.__init__
        rtacdata_copy.__init__ = GBData.__init__
        rtacdata_copy.early_rtac_copy = GBData.early_rtac_copy

        return rtacdata_copy(scenario, rtacdata_init, **kwargs)

    else:
        return rtacdata(scenario, **kwargs)


if __name__ == "__main__":
    pass
