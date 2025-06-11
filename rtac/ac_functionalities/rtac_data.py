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
    ReACTR = 1
    ReACTRpp = 2
    CPPL = 3


class Distribution(Enum):
    uniform = 1
    log = 2


class ValType(Enum):
    str = 1
    int = 2


class ParamType(Enum):
    discrete = 1
    continuous = 2
    categorical = 3
    binary = 4


@dataclass
class DiscreteParameter:
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
    paramtype: str
    flag: bool = False
    valtype: ValType = ValType.str
    default: Optional[str | int] = None
    minval: Optional[int] = None
    maxval: Optional[int] = None
    values: list[str | int] = field(default_factory=list)


@dataclass
class BinaryParameter:
    paramtype: str
    default: Optional[str | int]
    valtype: ValType = ValType.int
    values: list[str | int] = field(default_factory=list)


class Parameter(Enum):
    discrete = DiscreteParameter
    continuous = ContinuousParameter
    categorical = CategoricalParameter
    binary = BinaryParameter


class Generator(Enum):
    default = 0
    random = 1
    crossover = 2
    cppl = 3


@dataclass
class Configuration:
    id: UUID
    conf: dict
    features: list
    gen: Generator
    gen_tourn: int


class TARunStatus(Enum):
    running = 1
    finished = 2
    capped = 3
    terminated = 4
    timeout = 5
    awaiting_start = 6


@dataclass
class TARun:
    config_id: str
    config: dict
    res: int | float
    time: float
    status: TARunStatus


@dataclass
class TournamentStats:
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
    increase = 1
    decrease = 2


class AbstractRTACData(ABC):
    """Abstract class to handle picklable data structures needed to coordinate
    and process tournaments."""

    @abstractmethod
    def __init__(self, scenario: argparse.Namespace) -> None:
        """Initialize all data structures needed for the RTAC."""


class RTACData(AbstractRTACData):
    """Class to handle picklable data structures needed to coordinate
    and process tournaments of the ReACTR implementation."""

    def __init__(self, scenario: argparse.Namespace, **kwargs) -> None:
        """Initialize all data structures needed for ReACTR tournaments.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        """
        huge_res = sys.float_info.max * 1e-100
        self.ev = Event()
        freeze_support()
        # Using int as flags (event), since ctypes do not allow for
        # enum objects.
        self.tournID = 0
        self.cores_start = Manager().list(
            [core for core in range(scenario.number_cores)])
        self.early_start_tournament = False
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
    """Class to handle picklable data structures needed to coordinate
    and process tournaments of the ReACTR++ implementation."""

    def __init__(self, scenario: argparse.Namespace, **kwargs) -> None:
        RTACData.__init__(self, scenario)
        """Initialize additional data structures needed for ReACTR++
        tournaments.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
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
    """Class to handle picklable data structures needed to coordinate
    and process tournaments of the Gray-Box implementation."""

    def __init__(self, scenario: argparse.Namespace, rtacdata_init, **kwargs) -> None:
        rtacdata_init(self, scenario, **kwargs)
        """Initialize additional data structures needed for Gray-Box
        tournaments.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        """
        self.rec_data = {core: Manager().dict()
                         for core in range(scenario.number_cores)}

        self.RuntimeFeatures = Manager().list(
            [[] for core in range(scenario.number_cores)])


def rtacdata_factory(scenario: argparse.Namespace, **kwargs) \
        -> AbstractRTACData:
    """Class factory to return the initialized class with data structures
    appropriate to the RTAC method scenario.ac.

    :param scenario: Namespace containing all settings for the RTAC.
    :type scenario: argparse.Namespace
    :returns: Inititialized AbstractRTACData object matching the RTAC method of
        the scenario.
    :rtype: AbstractRTACData
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

        return rtacdata_copy(scenario, rtacdata_init, **kwargs)

    else:
        return rtacdata(scenario, **kwargs)


if __name__ == "__main__":
    pass
