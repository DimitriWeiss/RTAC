"""In this module results of the tournaments are processed and computations for
the next tournament ar performed."""

from abc import ABC, abstractmethod
from ac_functionalities.config_gens import DefaultConfigGen, RandomConfigGen
from ac_functionalities.ranking import trueskill, cppl
from ac_functionalities.rtac_data import (
    RTACData,
    Configuration,
    ACMethod,
    InterimMeaning
)
from ac_functionalities.logs import RTACLogs
from multiprocessing import Value
import argparse
import random
import sys
import numpy as np
import uuid
import copy
from scipy.stats import rankdata


class contender(object):
    """Helper object for setting trueskill ranks."""
    pass


class AbstractResultProcessing(ABC):
    """Abstract class with functions to process tournament results."""

    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs) -> None:
        """Initialize tournament result processing class.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        :param logs: Object containing loggers and logging functions.
        :type: RTACLogs
        """
        self.scenario = scenario
        self.logs = logs
        self.default_config_gen = DefaultConfigGen(self.scenario)
        self.random_config_gen = RandomConfigGen(self.scenario)
        self.tourn_nr = 0
        self.contender_dict = {}
        self.huge_float = sys.float_info.max * 1e-100
        self.pool = {}
        self.init_data()

    def init_data(self) -> dict[str: Configuration]:
        """Initialize tournament result processing data according to ReACTR
        implementation.

        :returns: Randomly selected contenders.
        :rtype: dict
        """
        if self.scenario.resume:
            # Data is loaded instead of intialized.
            pass
        elif self.scenario.pws:
            # Initialize pool of contender configurations incl. default
            default_config = self.default_config_gen.generate()
            self.pool[default_config.id] = default_config
            for _ in range(self.scenario.contenders - 1):
                random_config = self.random_config_gen.generate()
                self.pool[random_config.id] = random_config

            # Randomly initialize contender dict of first tournament
            # incl. default
            self.contender_dict[default_config.id] = default_config
            random_pick = random.sample(list(self.pool.values())[1:],
                                        self.scenario.number_cores - 1)
        else:
            # Initialize pool of cntender configurations
            for _ in range(self.scenario.contenders):
                random_config = self.random_config_gen.generate()
                self.pool[random_config.id] = random_config

            # Randomly initialize contender dict of first tournament
            random_pick = random.sample(list(self.pool.values()),
                                        self.scenario.number_cores)

        for rp in random_pick:
            self.contender_dict[rp.id] = rp

    @abstractmethod
    def process_results(self, rtac_data: RTACData) -> None:
        """Perform tournament result processing necessary to replace contenders
        in pool and select contenders for next tournament/problem instance.

        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        """

    @abstractmethod
    def manage_pool(self) -> None:
        """Replace contenders in pool if necessary."""

    @abstractmethod
    def select_contenders(self) -> None:
        """Select contenders for next tournament/problem instance."""

    def process_tourn(self, rtac_data: RTACData) -> str:
        """Manage result processing.

        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        :returns: ID of the configuration to have won the previous tournament
        :rtype: str
        """

        if not self.scenario.baselineperf:
            self.process_results(rtac_data)
            if self.rtac_data.winner.value != 0:
                self.manage_pool()
            self.select_contenders()
        else:
            self.rtac_data = rtac_data
            self.rtac_data.newtime = self.rtac_data.ta_res_time[0]

        if self.scenario.verbosity in (1, 2):
            if self.rtac_data.winner.value == 0:
                winner = None
            else:
                winner = self.rtac_data.winner.value
            print('\n\n')
            print('Winner was', winner)
            print('\n\n')

        return self.rtac_data.winner.value

    def get_contender_dict(self) -> dict[str: Configuration]:
        """Returns contender_dict.

        :returns: Configuration selected to run in next
        tournament/on next problem instance. Dictionary with configuration id
        as key and Configuration object as value.
        :rtype: dict[str: Configuration]
        """
        return self.contender_dict


class ResultProcessing(AbstractResultProcessing):
    """Processes results of previous tournament."""

    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs) -> None:
        """Initialize tournament result processing class as necessary for
        ReACTR implementation.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        :param logs: Object containing loggers and logging functions.
        :type: RTACLogs
        """
        super().__init__(scenario, logs)
        self.init_scores()

    def init_scores(self) -> None:
        """Initialize scores dict for trueskill."""
        self.scores = dict.fromkeys(list(self.pool.keys()), 
                                        (trueskill.INITIAL_MU,
                                         trueskill.INITIAL_SIGMA))

    def find_best(self, current_dict: list, number: int) -> list:
        """Find the configurations scored as best by trueskill.

        :param current_dict: Dictionary with confguration ids as key and
            trueskill scores (mu, sigma) as value.
        :type current_dict: dict
        :param current_dict: Number of best configurations to draw.
        :type current_dict: int
        :returns: List of best configurations.
        :rtype: list
        """
        best_list = sorted(current_dict.items(), key=lambda kv: kv[1][1])

        return best_list[:number]

    def get_winner(self, times: list[float], res: list[float]) \
            -> tuple[int, list[int]]:
        """Get index of the winning configuration and ranks.

        :param times: List of time results of previous tournament.
        :type times: list[float]
        :param res: List of objective results of previous tournament.
        :type res: list[float]
        :returns: Index of winner and ranks.
        :rtype: tuple[int, list[int]]
        """
        if not self.scenario.objective_min:
            winner = times.index(min(times))

        else:
            winner = res.index(min(res))

        ranks = []
        for core in range(self.scenario.number_cores):
            if core == winner:
                ranks.append(1)
            else:
                ranks.append(2)

        return winner, ranks

    def process_results(self, rtac_data: RTACData) -> None:
        """Perform tournament result processing necessary to replace contenders
        in pool and select contenders for next tournament/problem instance
        according to ReACTR implementation.

        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        """
        self.rtac_data = rtac_data
        for core in range(self.scenario.number_cores):
            # Setting timeout as time result if ta was terminated
            if self.rtac_data.ta_res[core] == self.huge_float:
                self.rtac_data.ta_res_time[core] = self.scenario.timeout

        times = list(self.rtac_data.ta_res_time[:])
        self.rtac_data.newtime = min(times)
        res = list(self.rtac_data.ta_res[:])
        self.rtac_data.best_res = min(res)

        winner, ranks = self.get_winner(times, res)

        if self.scenario.verbosity in (1, 2):
            res = list(self.rtac_data.ta_res[:])
            tourn_results = \
                [self.contender_dict, [[r, t] for r, t in zip(res, times)]]

            print('\nResults of this tournament:\n \nContender', ' ' * 19,
                  '   [Objective,Time]')
            for a in zip(*tourn_results):
                print(*a)
            print('\n')
            tr_str = ''
            for tr in tourn_results:
                tr_str += str(tr) + ' '
            self.logs.general_log(f'Results of this tournament: {tr_str}')

        # Set the results of the tournament
        individuals = [None] * self.scenario.number_cores
        contender_ids = list(self.contender_dict.keys())
        for core in range(self.scenario.number_cores):
            skill = (self.scores[contender_ids[core]][0], 
                     self.scores[contender_ids[core]][1])
            individuals[core] = contender()
            individuals[core].skill = skill
            individuals[core].rank = ranks[core]

            if self.scenario.verbosity == 2:
                print('Contender', contender_ids[core], 'has the rank',
                      individuals[core].rank)
        print('\n')

        # Process the results of the tournament
        trueskill.AdjustPlayers(individuals)

        if self.scenario.verbosity in (1, 2):
            print('\nSkills of the contenders from tournament:\n \nContender',
                  ' ' * 31, '   (Mu', ' ' * 14, ', Sigma)')

        # Update Scores
        for core in range(self.scenario.number_cores):
            self.scores[contender_ids[core]] = individuals[core].skill
            if self.scenario.verbosity in (1, 2):
                print(contender_ids[core], 'skills are:',
                      individuals[core].skill)

    def manage_pool(self) -> None:
        """Replace contenders in pool according to Mu and Sigma (TrueSKill)."""
        # Replace contenders in self.pool, if performance below average
        contender_ids = list(self.pool.keys())
        for core in range(self.scenario.contenders):
            # Contenders with a performance variance (sigma) <=
            # self.scenario.kill are eligible to be replaced
            if self.scores[contender_ids[core]][1] <= self.scenario.kill: 
                tournament_list = {}
                names = list(self.scores)
                for c in range(self.scenario.contenders):
                    tournament_list[c] = [names[c], self.scores[names[c]][0]]

                # Sort according to mean performance (Mu)
                best_list = \
                    self.find_best(tournament_list, self.scenario.contenders)

                # Get 5 best performing contenders for breeding
                best_five = best_list[len(best_list) - 5:]

                # Contenders which also have mean performance lower
                # than median performane are replaced by new contenders
                if self.scores[contender_ids[core]][0] \
                        < self.scores[best_list[
                            int(self.scenario.contenders / 2)][1][0]][0]:       

                    # Replace by randomly generated contender if chance
                    # is lower than self.scenario.chance
                    chance = np.random.uniform(1, 100, 1)
                    mutated_individual = self.random_config_gen.generate()
                    if chance <= self.scenario.chance:
                        del self.pool[contender_ids[core]]
                        new_contender_id = mutated_individual.id
                        self.pool[new_contender_id]\
                            = mutated_individual
                        if self.scenario.verbosity in (1, 2):
                            print('\nReplaced contender',
                                  f'{contender_ids[core]} by randomly',
                                  'generated contender.')

                    # Else generate new contender by genetic crossover
                    elif chance > self.scenario.chance:
                        mutated = 0
                        parent_one, parent_two = \
                            random.sample([0, 1, 2, 3, 4], 2)
                        self.scenario.config_space
                        del self.pool[contender_ids[core]]
                        new_contender_id = uuid.uuid4().hex
                        self.pool[new_contender_id] = {}

                        for param in self.scenario.config_space:
                            which = np.random.uniform(0, 1, 1)

                            if 0.5 < which:
                                self.pool[new_contender_id][param] \
                                    = self.pool[
                                    best_five[parent_one][1][0]].conf[param]

                            elif 0.5 >= which:
                                self.pool[new_contender_id][param] \
                                    = self.pool[
                                    best_five[parent_two][1][0]].conf[param]

                            mutation = int(np.random.uniform(0, 100, 1))
                            if mutation <= self.scenario.mutate:
                                self.pool[new_contender_id][param]\
                                    = mutated_individual.conf[param]
                                mutated = mutated + 1

                        self.pool[new_contender_id] = \
                            Configuration(
                                new_contender_id,
                                self.pool[new_contender_id], [])
                        if self.scenario.verbosity in (1, 2):
                            print('\nReplaced contender',
                                  f'{contender_ids[core]} by contender',
                                  'generated via crossover.')
                        if self.scenario.verbosity == 2:
                            print(f'Mutation of {mutated} genes happened for',
                                  f'the new contender {new_contender_id}!\n')

                    # Delete scores of replaced contender and insert initial
                    # scores for new contender
                    del self.scores[contender_ids[core]]
                    self.scores[new_contender_id] = \
                        (trueskill.INITIAL_MU, trueskill.INITIAL_SIGMA)

    def select_contenders(self) -> None:
        """Select scenario.contenders == #contenders for next
        tournament/problem instance: top number of 
        'self.scenario.keeptop' and the rest randomly."""
        tournament_list = {}

        # Choose the two best contenders
        names = list(self.scores)
        for c in range(self.scenario.contenders):
            tournament_list[c] = [names[c], self.scores[names[c]][0]]
        best_list = self.find_best(tournament_list, self.scenario.contenders)
        self.contender_dict = {}
        for keep in range(self.scenario.keeptop):
            contender_id = best_list[self.scenario.contenders - 1 - keep][1][0]
            self.contender_dict[contender_id] = self.pool[contender_id]

        # Fill in the rest with randomly chosen contenders from pool
        temp_pool = copy.copy(self.pool)
        for contender in self.contender_dict:
            del temp_pool[contender]
        random_pick = \
            random.sample(
                list(temp_pool.keys()),
                self.scenario.number_cores - self.scenario.keeptop)
        for rp in random_pick:
            self.contender_dict[rp] = temp_pool[rp]

        if self.scenario.verbosity == 2:
            print('\nNew contender list is:',
                  *self.contender_dict, '\n', sep='\n')


class ResultProcessingpp(ResultProcessing):
    """Process results of prvious tournament."""

    def duplicates(self, ranks: list[int], rank: float) -> list[int]:
        """List the indices of the result in the results list.

        :param ranks: List of objective results of previous tournament.
        :type ranks: list[int]
        :param rank: a single result.
        :type rank: float
        :returns: list of indices of this result in the results list.
        :rtype: list[int]
        """
        return [i for i, x in enumerate(ranks) if x == rank]

    def get_winner(self, times: list[float], res: list[float]) \
            -> tuple[int, list[int]]:
        """Get index of the winning configuration including last known
        intermidiate outputs to break ties. Additionally output complete
        ranking to compute more detailed assessment with trueskill.

        :param times: List of time results of previous tournament.
        :type times: list[float]
        :param res: List of objective results of previous tournament.
        :type res: list[float]
        :returns: Index of winner and ranks.
        :rtype: tuple[int, list[int]]
        """
        if not self.scenario.objective_min:
            winner = times.index(min(times))

        else:
            winner = res.index(min(res))

        interim_sorted = [[self.rtac_data.interim[j][i]
                          for j in range(self.scenario.number_cores)]
                          for i, _ in enumerate(self.rtac_data.interim[0])]

        interim_sorted = np.array(interim_sorted)
        interim_sorted = interim_sorted.astype(float)

        for i, isort in enumerate(interim_sorted):
            if self.rtac_data.interim_meaning[i] is \
                    InterimMeaning.decrease:
                interim_sorted[i] = rankdata(isort,
                                             method='dense',
                                             nan_policy="propagate")
            elif self.rtac_data.interim_meaning[i] is \
                    InterimMeaning.increase:
                interim_sorted[i] = rankdata([-1 * i if i is not None
                                              else None for i in isort],
                                             method='dense',
                                             nan_policy="propagate")

        ranks = [0 for core in range(self.scenario.number_cores)]

        for _, isort in enumerate(interim_sorted):
            for r, _ in enumerate(ranks):
                ranks[r] += isort[r]

        if self.scenario.objective_min:
            res_ranks = rankdata(res, method='dense', nan_policy="propagate")

            duplicates = []

            for rank in sorted(set(res_ranks)):
                duplicates.append(self.duplicates(res_ranks, rank))
            
            for duplicate in duplicates:
                if len(duplicate) > 1:
                    interim_ranks = [ranks[dup] for dup in duplicate]
                    tie_winner = \
                        duplicate[interim_ranks.index(min(interim_ranks))]
                    interim_ranks = rankdata(interim_ranks, method='dense')
                    interim_ranks -= min(interim_ranks)
                    for d, ir in zip(duplicate, interim_ranks):
                        for r, _ in enumerate(res_ranks):
                            if d == r and r != winner and r != tie_winner:
                                res_ranks[r] += interim_ranks[ir]
                            else:
                                res_ranks[r] += max(interim_ranks)

            ranks = res_ranks

        ranks = rankdata(ranks, method='dense')

        ranks[winner] = 0

        return winner, ranks


class ResultProcessingCPPL(AbstractResultProcessing):
    """Process results of previous tournament."""

    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs) -> None:
        """Initialize tournament result processing class as necessary for
        CPPL implementation.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        :param logs: Object containing loggers and logging functions.
        :type: RTACLogs
        """
        super().__init__(scenario, logs)
        self.init_data()
        self.cppl = cppl.CPPL(self.scenario, self.pool)

    def process_results(self, rtac_data: RTACData) -> None:
        """Perform tournament result processing necessary to replace contenders
        in pool and select contenders for next tournament/problem instance.

        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        """
        self.cppl.process_results(rtac_data)

    def manage_pool(self) -> None:
        """Replace contenders in pool if necessary."""
        self.cppl.manage_pool()

    def select_contenders(self) -> None:
        """Select contenders for next tournament/problem instance."""
        self.contender_dict = cppl.select_contenders()

        if self.scenario.verbosity == 2:
            print('\nNew contender list is:',
                  *self.contender_dict, '\n', sep='\n')


'''
class ResultProcessingGB(ResultProcessingCPPL):
    """Process results of previous tournament."""

    def __init__(self, scenario: argparse.Namespace, logs: RTACLogs) -> None:
        """Initialize tournament result processing class as necessary for
        CPPL implementation.

        :param scenario: Namespace containing all settings for the RTAC.
        :type scenario: argparse.Namespace
        :param logs: Object containing loggers and logging functions.
        :type: RTACLogs
        """
        super().__init__(scenario, logs)
        self.init_data()
        self.cppl = cppl.CPPL(self.scenario)
        self.model = costcla.init()

    def process_results(self, rtac_data: RTACData) -> None:
        """Perform tournament result processing necessary to replace contenders
        in pool and select contenders for next tournament/problem instance.

        :param rtac_data: Object containing data and objects necessary
            throughout the rtac modules.
        :type rtac_data: RTACData
        """
        self.model = model.fit(training_data, training_y)
        self.cppl.process_results(rtac_data)
'''


def processing_factory(scenario, logs) -> AbstractResultProcessing:
    """Class factory to return the initialized class with data structures
    appropriate to the RTAC method scenario.ac.

    :param scenario: Namespace containing all settings for the RTAC.
    :type scenario: argparse.Namespace
    :returns: Inititialized BaseTARunner object matching the RTAC method of
        the scenario.
    :rtype: BaseTARunner
    """
    if scenario.ac is ACMethod.ReACTR:
        return ResultProcessing(scenario, logs)

    elif scenario.ac is ACMethod.ReACTRpp:
        return ResultProcessingpp(scenario, logs)

    elif scenario.ac is ACMethod.CPPL:
        return ResultProcessingCPPL(scenario, logs)

    '''

    elif scenario.ac == ACMethod.GRAYBOX:
        return ResultProcessingGB(scenario, logs)
    '''


if __name__ == "__main__":
    pass
