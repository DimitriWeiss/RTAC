"""This module contains functions for configuration generation."""

from abc import ABC, abstractmethod
import argparse
import sys
import uuid
import random
import math
import numpy as np
from ac_functionalities.rtac_data import (
    Configuration,
    ParamType,
    ValType,
    Distribution,
    DiscreteParameter,
    ContinuousParameter,
    CategoricalParameter,
    BinaryParameter
)


class AbstractConfigGen(ABC):
    """Abstract class for generation of configurations."""

    @abstractmethod
    def __init__(self, scenario: argparse.Namespace):
        """Initialize configuration generation class."""

    @abstractmethod
    def generate(self) -> Configuration:
        """Generates and returns configuration."""


class DefaultConfigGen(AbstractConfigGen):
    """Generate default configuration."""

    def __init__(self, scenario: argparse.Namespace):
        """Initialize default configuration class.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        """
        self.config_space = scenario.config_space
        self.default_config = {}

    def generate(self) -> Configuration:
        """Generates and stores default configuration.

        :returns: Default configuration.
        :rtype: ac_functionalities.rtac_data.Configuration
        """
        if not self.default_config:
            default_config = {}
            for param, definition in self.config_space.items():
                if definition.default is not None:
                    default_config[param] = definition.default
                else:  # If default not defined, define naively
                    if definition.paramtype in \
                            (ParamType.discrete, ParamType.continuous):
                        default_config[param] = \
                            (definition.maxval - definition.minval) / 2
                    else:
                        default_config[param] = definition.values[0]

            self.default_config = \
                Configuration(uuid.uuid4().hex, default_config, [])

            return self.default_config
        else:
            return self.default_config


class RandomConfigGen(AbstractConfigGen):
    """Generate random configuration."""

    def __init__(self, scenario: argparse.Namespace):
        """Initialize random configuration class.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        """
        self.config_space = scenario.config_space

    def generate(self) -> Configuration:
        """Generates random configuration.

        :returns: Random configuration.
        :rtype: ac_functionalities.rtac_data.Configuration
        """
        random_config = {}
        for param_name, parameter in self.config_space.items():

            if isinstance(parameter, BinaryParameter):
                if isinstance(parameter.valtype, ValType.int):
                    random_config[param_name] = random.choice([0, 1])
                elif isinstance(parameter.valtype, ValType.str):
                    random_config[param_name] = random.choice(parameter.values)

            elif isinstance(parameter, CategoricalParameter):
                if parameter.valtype is ValType.int:
                    if parameter.values:
                        random_config[param_name] = \
                            random.choice(parameter.values)
                    else:
                        random_config[param_name] = \
                            random.choice(list(range(
                                parameter.minval, parameter.maxval)))
                elif parameter.valtype is ValType.str:
                    if not parameter.flag:
                        random_config[param_name] = \
                            random.choice(parameter.values)
                    else:
                        random_config[random.choice(parameter.values)] = ''              

            elif isinstance(parameter,
                            (ContinuousParameter, DiscreteParameter)):
                minval = parameter.minval
                maxval = parameter.maxval

                if parameter.distribution is Distribution.uniform:
                    random_config[param_name] = random.uniform(minval, maxval)

                elif parameter.distribution is Distribution.log:
                    tiny_float = sys.float_info.min
                    probabpos = np.float64(round(parameter.probabpos, 3))
                    if parameter.includezero:
                        probabzero = \
                            np.float64(round(parameter.probabilityzero, 3))
                        probabneg = \
                            np.float64(round(1 - probabpos - probabzero, 3))
                    else:
                        probabneg = np.float64(round(1 - probabpos, 3))

                    values = []
                    weights = []

                    if parameter.splitbydefault and parameter.default != 0:
                        split_high = math.log(parameter.default)
                        split_low = math.log(parameter.default - tiny_float)
                    else:
                        if parameter.logonpos:
                            split_high = tiny_float
                        else:
                            split_high = minval
                        if parameter.logonneg:
                            split_low = -tiny_float
                        else:
                            split_low = maxval

                    if parameter.logonpos:
                        logmaxval = math.log(maxval)
                        values.append(
                            math.exp(np.random.uniform(
                                split_high, logmaxval, size=1)))
                    else:
                        values.append(random.uniform(split_high, maxval))

                    weights.append(probabpos)

                    if parameter.logonneg:
                        if minval > 0:
                            logminval = math.log(minval)
                            values.append(
                                math.exp(np.random.uniform(
                                    split_low, logminval, size=1)))
                        elif minval == 0:
                            logminval = math.log(tiny_float)
                            values.append(
                                math.exp(np.random.uniform(
                                    split_low, logminval, size=1)))
                        elif minval < 0:
                            logminval = math.log(-minval)                        
                            values.append(
                                -math.exp(np.random.uniform(
                                    split_low, -logminval, size=1)) - minval)
                    else:
                        values.append(random.uniform(minval, split_low))

                    weights.append(probabneg)
                    
                    if parameter.includezero:
                        values.append(0)
                        weights.append(probabzero)

                    random_config[param_name] = \
                        np.random.choice(values, 1, p=weights)[0]

                if isinstance(parameter, DiscreteParameter):
                    random_config[param_name] = \
                        int(round(random_config[param_name]))

        return Configuration(uuid.uuid4().hex, random_config, [])
