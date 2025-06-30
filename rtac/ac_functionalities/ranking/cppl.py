"""Implements the CPPL class for RTAC."""

import random
import uuid
import copy
import argparse
import importlib
from collections import Counter
from itertools import islice
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters.categorical import CategoricalHyperparameter
from rtac.ac_functionalities.rtac_data import (
    Configuration,
    ParamType,
    Generator,
    ValType
)
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
    normalize
)
from threadpoolctl import threadpool_limits
from concurrent.futures import ThreadPoolExecutor, as_completed


class CPPL():
    """
    CPPL Bandit and AC functions.

    Note
    ----

    Implementation based on the paper: "Pool-based realtime algorithm
    configuration: A preselection bandit approach"

    Parameters
    ----------
    scenario : argparse.Namespace
        Namespace containing all settings for the RTAC.
    pool : dict
        Pool of configurations.
    contender_dict : dict
        Dictionary containing Configurations that ran in last tournament.
    """

    def __init__(self, scenario: argparse.Namespace, pool: dict,
                 contender_dict: dict = None):
        self.time_sum = 0
        self.scenario = scenario
        self.keeptop = self.scenario.keeptop
        config_space = scenario.config_space
        del self.scenario.config_space
        self.pool = copy.deepcopy(pool)
        self.pool_size = len(self.pool)
        self.random_config_gen = None
        self.transformed_pool = {}
        if contender_dict is None:
            self.contender_dict = {}
        else:
            self.contender_dict = contender_dict
        self.prior_contender_dict = {}
        self.features = []
        self.standard_scaler = StandardScaler()
        if self.scenario.online_instance_train:
            self.pca_obj_inst = \
                IncrementalPCA(n_components=self.scenario.nc_pca_f,
                               batch_size=1)
            self.pre_train(self.scenario.instance_pre_train)
        else:
            self.pca_obj_inst = PCA(n_components=self.scenario.nc_pca_f)
            self.pre_train(self.scenario.instance_pre_train)
        
        fg_module = importlib.import_module(self.scenario.feature_gen)
        fg_name = self.scenario.feature_gen_name
        self.feature_gen = \
            getattr(fg_module, fg_name)(self.scenario.feature_path)

        # Calibrate MinMaxScaler for non-categorical parameters
        self.init_param_scaler(config_space)
        del config_space

        # Initialize One-Hot-Encoder
        self.init_onehot_encoder()

        pool_items = copy.deepcopy(self.pool).items()

        for conf_id, conf in pool_items:
            self.transformed_pool[conf_id] = self.transform_conf(conf)

        self.pca_obj_params = PCA(n_components=self.scenario.nc_pca_p)
        self.transformed_pool_array = \
            np.asarray(list(self.transformed_pool.values()))
        self.pca_obj_params.fit(self.transformed_pool_array)
        self.transformed_pool_array = \
            self.pca_obj_params.transform(self.transformed_pool_array)
         
        self.dim = self.compute_array_dimension(self.scenario.nc_pca_f,
                                                self.scenario.nc_pca_p)

        self.theta_hat = np.zeros(self.dim, dtype=np.float32)
        self.theta_bar = self.theta_hat
        self.len_theta_bar = len(self.theta_bar)

        self.grad_op_sum = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.hess_sum = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.omega = self.scenario.omega
        self.gamma_1 = self.scenario.gamma
        self.alpha = self.scenario.alpha
        self.t = 1
        self.Y_t = 0
        self.S_t = []
        self.S_t_prior = []
        self.grad = np.zeros(self.dim, dtype=np.float32)

        self.pool_replacement = False
        self.tourn = 0

        self.win_decay = np.zeros(self.pool_size)

        self.Sigma_bar = np.identity(len(self.theta_bar)) * 1.0

        self.record_bandit()

    def compute_array_dimension(self, nc_pca_f: int,
                                nc_pca_p: int) -> int:
        """Compute the correct dimension of np.ndarrays used in bandit.

        Parameters
        ----------
        nc_pca_f : int
            Dimension of PCA for problem instance features.
        nc_pca_p : int
            Dimension of PCA for Configuration values.

        Returns
        -------
        int
            Dimension of np.ndarrays used in bandit.
        """
        if self.scenario.jfm in ('concatenation', 'kronecker'):
            d = nc_pca_f + nc_pca_p
        elif self.scenario.jfm == 'polynomial':
            d = 4
            for i in range((
                    nc_pca_f + nc_pca_p) - 2):
                d = d + 3 + i
        return d

    def record_bandit(self) -> None:
        """
        Records bandit state to later log it.

        Returns
        -------
        None
        """
        self.bandit = {'theta_hat': np.array2string(self.theta_hat,
                                                    threshold=np.inf),
                       'theta_bar': np.array2string(self.theta_bar,
                                                    threshold=np.inf),
                       'grad_op_sum': np.array2string(self.grad_op_sum,
                                                      threshold=np.inf),
                       'hess_sum': np.array2string(self.hess_sum,
                                                   threshold=np.inf),
                       't': self.t,
                       'Y_t': self.Y_t,
                       'S_t': self.S_t,
                       'grad': np.array2string(self.grad, threshold=np.inf)}

        if hasattr(self, 'X_t'):
            self.bandit['X_t'] = np.array2string(self.X_t, threshold=np.inf)
        if hasattr(self, 'hess'):
            self.bandit['hess'] = np.array2string(self.hess, threshold=np.inf)

        if self.tourn == 0:
            self.bandit_models = {'standard_scaler': self.standard_scaler,
                                  'min_max_scaler': self.min_max_scaler,
                                  'one_hot_encoder': self.one_hot_enc,
                                  'pca_obj_params': self.pca_obj_params,
                                  'pca_obj_inst': self.pca_obj_inst}
        elif self.scenario.online_instance_train:
            self.bandit_models = {'standard_scaler': self.standard_scaler,
                                  'pca_obj_inst': self.pca_obj_inst}
        else:
            self.bandit_models = {}

    def process_results(self) -> None:
        """
        Processing results of the last tournament.
        Returns
        -------
        None
        """

        self.update_data()

        self.grad = self.gradient()
        self.grad_op_sum = self.grad_op_sum + np.outer(self.grad, self.grad)

        with threadpool_limits(limits=1):
            self.theta_hat = \
                self.theta_hat + (
                    self.gamma_1 * self.t ** (-self.alpha) * self.grad
                )
            # Forgetting factor update block
            x = self.X_t[self.Y_t, :]

            # Compute theta_bar with forgetting factor
            Sigma_inv = np.linalg.inv(self.Sigma_bar)
            Sigma_inv = \
                self.scenario.forgetting_factor * Sigma_inv + (
                    (1.0 / self.scenario.obs_noise) * np.outer(x, x)
                )
            self.Sigma_bar = np.linalg.inv(Sigma_inv)
            self.theta_bar = self.Sigma_bar @ (
                self.scenario.forgetting_factor * Sigma_inv @
                self.theta_bar + (
                    1.0 / self.scenario.obs_noise
                ) * x * self.scenario.cppl_reward
            )

            self.hess = self.hessian()
            self.hess_sum = self.hess_sum + self.hess
            
            self.skill_and_confidence()

            self.record_bandit()

    def update_data(self) -> None:
        """
        Update data used in bandit computations by tournament results.

        Returns
        -------
        None
        """

        with threadpool_limits(limits=1):

            self.t += 1

            # Get features from new configurations in pool, if pool changed
            if self.pool_replacement:
                self.transformed_pool = {}
                for conf_id, conf in copy.deepcopy(self.pool).items():
                    self.transformed_pool[conf_id] = self.transform_conf(conf)
                self.transformed_pool_array = \
                    np.asarray(list(self.transformed_pool.values()))
                self.transformed_pool_array = \
                    self.pca_obj_params.transform(self.transformed_pool_array)

            # Get instance features
            new_inst_features = self.feature_gen.get_features(self.instance)
            if self.scenario.online_instance_train:        
                self.standard_scaler.partial_fit(new_inst_features)
                self.scaled_features = \
                    self.standard_scaler.transform(new_inst_features)[0]
                self.pca_obj_inst.partial_fit(self.features)
            else:
                self.scaled_features = \
                    self.standard_scaler.transform(new_inst_features)[0]
            self.scaled_features = \
                self.pca_obj_inst.transform(
                    self.scaled_features.reshape(1, -1)
                )

            results = self.results

            # Get pool index of winner
            self.Y_t = min(range(len(results)), key=results.__getitem__)
            contender_ids = list(copy.deepcopy(self.pool).keys())
            self.S_t = \
                [contender_ids.index(k) for k in self.contender_dict.keys()]
            self.Y_t = self.S_t[self.Y_t]

        self.context_specific_feature_matrix()

    def context_specific_feature_matrix(self) -> None:
        """
        Compute the context-specific feature matrix.

        Returns
        -------
        None
        """
        with threadpool_limits(limits=1):
            self.X_t = np.zeros((self.pool_size, len(self.theta_bar)),
                                dtype=np.float32)
            for i in range(self.pool_size):
                self.X_t[i, :] = self.joinFeatureMap(
                    self.transformed_pool_array[i,], 
                    self.scaled_features[0],
                    self.scenario.jfm
                )

            normalize(self.X_t, norm='max', copy=False)

    def gradient(self) -> np.ndarray:
        """
        Computes gradient learnt from results.

        Returns
        -------
        np.ndarray
            Gradient learnt from all previous results.
        """
        with threadpool_limits(limits=1):
            d = 0
            n = np.zeros(self.len_theta_bar)
            for ll in self.S_t:
                d = d + np.exp(np.dot(self.theta_hat, self.X_t[ll, :]))
                n = n + (self.X_t[ll, :] * np.exp(
                    np.dot(self.theta_hat, self.X_t[ll, :])))

            return self.X_t[self.Y_t, :] - (n / d)

    def hessian(self) -> np.ndarray:
        """
        Computes Hessian matrix expressing confidence in skills.

        Returns
        -------
        np.ndarray
            Hessian matrix expressing confidence in skills.
        """
        with threadpool_limits(limits=1):

            t_1 = np.zeros((self.len_theta_bar))
            for ll in self.S_t:
                t_1 = t_1 + (self.X_t[ll, :] * np.exp(
                    np.dot(self.theta_bar, self.X_t[ll, :])))
            n_1 = np.outer(t_1, t_1)

            n_2 = 0
            for j in self.S_t:
                n_2 = n_2 + (np.exp(
                    np.dot(self.theta_bar, self.X_t[j, :])) * np.outer(
                    self.X_t[j, :], self.X_t[j, :]))

            d = 0
            for ll in self.S_t:
                d = d + np.exp(
                    np.dot(self.theta_bar, self.X_t[ll, :]))

            return (n_1 / (d ** 2)) - (n_2 / d)

    def joinFeatureMap(self, x, y, mode) -> np.ndarray:
        """
        Joining Configuration values and problem instance features.

        Parameters
        ----------
        x : np.ndarray
            Configuration values.
        y: np.ndarray
            Problem instance features.
        mode: str
            Mode of combining Configuration values and instance features.

        Returns
        -------
        np.ndarray
            Combined Configuration values and problem instance features.
        """
        with threadpool_limits(limits=1):
            if mode == 'concatenation':
                return np.concatenate((x, y), axis=0) 
            elif mode == 'kronecker':
                return np.kron(x, y)
            elif mode == 'polynomial':
                poly = PolynomialFeatures(degree=2, interaction_only=True)
                return poly.fit_transform(
                    np.concatenate((x, y), axis=0).reshape(1, -1))

    def manage_pool(self) -> None:
        """
        Deciding if Configurations need to be replaced and generating new 
        ones if necessary.

        Returns
        -------
        None
        """
        self.insert_in_pool(self.generate_configs(self.discard_configs()))

    def discard_configs(self) -> int:
        """
        Assess the quality of the Configurations in the pool and decide which 
        ones to replace.

        Returns
        -------
        int
            Number of Configurations to be replaced.
        """

        self.asses_configurations()

        return len(self.discard)

    def crossover(self, parents: list[Configuration],
                  nr_children: int) -> list[Configuration]:
        """
        Perform genetic crossover on the provided Configurations to generate 
        offspring, with small probability of randomly generated Configurations 
        nr_children of times.

        Parameters
        ----------
        parents : list[Configuration]
            The Configurations to be used for genetic crossover.
        nr_children : int
            Number of how many offspring to generate.
        
        Returns
        -------
        list[Configuration]
            List of generated Configurations.
        """

        parent_a, parent_b = parents

        new_candids = []
        for child in range(nr_children):
            random_individual_probab = random.uniform(0, 1)
            random_individual = self.random_config_gen.generate(self.tourn)
            if random_individual_probab < self.scenario.chance / 100:
                new_candid = random_individual
            else:
                new_candid = {}
                for param_name, value in parent_a.items():
                    rn = random.uniform(0, 1)
                    mutate = random.uniform(0, 1)
                    if rn > 0.5:
                        new_candid[param_name] = parent_a[param_name]
                    else:
                        new_candid[param_name] = parent_b[param_name]
                    if mutate < self.scenario.mutate / 100:
                        new_candid[param_name] = \
                            random_individual.conf[param_name]
                new_candid = Configuration(
                    uuid.uuid4().hex, new_candid, [],
                    Generator.cppl, self.tourn
                )
            new_candids.append(new_candid)

        return new_candids

    def generate_configs(
            self, nr_discarded: int) -> dict[str, Configuration] | None:
        """
        Perform genetic crossover on the provided Configurations to generate 
        offspring, with small probability of randomly generated Configurations 
        on each available core. Asses generated Configurations with CPPL 
        model. Ensure no duplicates are inserted into the pool (possible due 
        to CPPL model preferring configuration values). Aim at providing 
        nr_discarding Configurations (if number is not reduced by omitting 
        duplicates).

        Parameters
        ----------
        nr_discarded : int
            Number of how many new Configurations to generate.
        
        Returns
        -------
        dict[str, Configuration] or None
            Dict of generated Configurations or None.
        """

        for _ in range(self.scenario.number_cores):
            # Choose the two configurations assessed as best as parents
            parents = \
                [copy.deepcopy(list(self.pool.values())[self.S_t[0]].conf),
                 copy.deepcopy(list(self.pool.values())[self.S_t[1]].conf)]

        configs = []
        best_candids = {}

        if nr_discarded > 0:
            self.pool_replacement = True
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(
                           self.crossover, parents,
                           nr_discarded * self.scenario.gen_mult)
                           for i in range(self.scenario.number_cores)]

                for future in as_completed(futures):
                    config_list = future.result()
                    configs.append(config_list)

            configs = [item for sublist in configs for item in sublist]

            pool_conf_settings = \
                [conf.conf for conf in copy.deepcopy(self.pool).values()]

            unique_configs = []
            new_list = []
            for idx, conf in enumerate(configs):
                if (conf.conf in 
                        pool_conf_settings or conf.conf in unique_configs):
                    break
                else:
                    unique_configs.append(conf.conf)
                    new_list.append(conf)
            configs = new_list

            existing_confs = \
                set(tuple(sorted(obj.conf.items())) for obj in
                    copy.deepcopy(self.pool).values())
            best_cand_confs_counter = \
                Counter(tuple(sorted(obj.conf.items())) for obj in configs)
            configs = [
                obj for obj in configs if best_cand_confs_counter[
                    tuple(sorted(obj.conf.items()))
                ] == 1
            ]

            new_list = []

            for obj in configs:
                conf_tuple = tuple(sorted(obj.conf.items()))
                if conf_tuple not in existing_confs:
                    new_list.append(obj)

            configs = new_list

            if configs:
                ctt = copy.deepcopy(configs)
                transformed_configs = \
                    [self.transform_conf(conf) for conf in ctt]
                transformed_configs = \
                    self.pca_obj_params.transform(transformed_configs)
                S_t = self.assess_children(
                    np.asarray(transformed_configs), nr_discarded
                )

                self.keeptop_gen = int(nr_discarded * (
                    self.scenario.keeptop / self.scenario.number_cores))

                if self.scenario.epsilon_greedy:
                    best_candids = {
                        configs[idx].id: configs[idx] for idx in 
                        S_t[:self.keeptop_gen]
                    }

                    best_candids_confs = \
                        [conf.conf for conf in best_candids.values()]
                    unique_configs = []
                    maxidx = len(configs) - 1
                    used_idx = [i for i in range(maxidx + 1)]
                    for s in S_t[:self.keeptop_gen]:
                        used_idx.remove(s)

                    if used_idx:
                        for i in range(nr_discarded - self.keeptop_gen):
                            ridx = random.choice(used_idx)
                            while (configs[ridx].conf in
                                   best_candids_confs and configs[ridx].conf in
                                   unique_configs):
                                used_idx.remove(ridx)
                                if not used_idx:
                                    break
                                ridx = random.choice(used_idx)
                                
                            best_candids[configs[ridx].id] = configs[ridx]
                            unique_configs.append(configs[ridx].conf)
                            if not used_idx:
                                break
                    else:
                        best_candids = \
                            {configs[idx].id: configs[idx] for idx in S_t}
                else:
                    best_candids = \
                        {configs[idx].id: configs[idx] for idx in S_t}

            if len(best_candids) < nr_discarded and len(best_candids) > 0:
                self.discard = self.discard[:len(best_candids)]
                return best_candids

            elif len(best_candids) == 0:
                self.pool_replacement = False
                return None

            elif len(best_candids) > len(self.discard):
                best_candids = \
                    dict(islice(best_candids.items(), 0, len(self.discard)))

            else:
                return best_candids
        else:
            self.pool_replacement = False
            return None

    def insert_in_pool(self, configs):
        if configs is not None:
            pool_list = list(copy.deepcopy(self.pool).values())

            for d in self.discard:
                del self.pool[pool_list[d].id]

            for conf_id, conf in configs.items():
                self.pool[conf_id] = conf

            print('\n')

            for c, d in zip(configs.values(), self.discard):
                disc_id = list(self.pool.keys())[d]
                if self.scenario.verbosity == 2:
                    print('Replaced contender', disc_id,
                          'by contender generated via', c.gen, '\n')

    def skill_and_confidence(self) -> None:
        """
        Compute the skill and confidence values for Configurations in pool 
        based on current feature matrix and save the indices of nr_cores best 
        of them to self.S_t.

        Returns
        -------
        None
        """
        with threadpool_limits(limits=1):

            self.S_t_prior = copy.deepcopy(self.S_t)

            # Estimated skill parameters
            self.v_hat = np.zeros(self.pool_size)
            for i in range(self.pool_size):
                self.v_hat[i] = \
                    np.exp(np.inner(self.theta_bar, self.X_t[i, :]))
            
            # Configuration quality estimation
            self.c_t = np.zeros(self.pool_size)
            try:
                V_hat = (1 / self.t) * self.grad_op_sum
                V_hat = V_hat.astype('float64')
                S_hat = (1 / self.t) * self.hess_sum
                S_hat = S_hat.astype('float64')
                try:
                    S_hat_inv = np.linalg.inv(S_hat)
                except Exception as e:
                    print(e)
                    S_hat_inv = np.linalg.pinv(S_hat)

                S_hat_inv = S_hat_inv.astype('float64')
                Sigma_hat = (1 / self.t) * np.dot(
                    np.dot(S_hat_inv, V_hat),
                    S_hat_inv)
                Sigma_hat_sqrt = sp.linalg.sqrtm(Sigma_hat)

                for i in range(self.pool_size):
                    M_i = np.exp(2 * np.dot(
                        self.X_t[i], self.theta_bar)
                    ) * np.outer(self.X_t[i], self.X_t[i])

                    self.c_t[i] = np.sqrt((
                        2 * np.log(self.t) + self.len_theta_bar + 2 * np.sqrt(
                            self.len_theta_bar * np.log(self.t))
                    ) * np.linalg.norm(
                        np.dot(np.dot(Sigma_hat_sqrt, M_i), Sigma_hat_sqrt)))

                if np.any(np.isinf(self.c_t)):
                    self.c_t = np.ones_like(self.c_t)

                elif max(self.c_t) != 0:
                    mc = max(self.c_t)
                    for i, c in enumerate(self.c_t):
                        if self.c_t[i] != 0:
                            self.c_t[i] = self.c_t[i] / mc

                self.v_hat = self.v_hat / max(self.v_hat)

                # Boost v_hat for recent winners
                self.win_decay *= self.scenario.win_decay
                self.win_decay[self.Y_t] += self.scenario.recent_winner_boost
                for i in range(self.pool_size):
                    self.v_hat[i] *= \
                        (1 + self.scenario.win_bonus * self.win_decay[i])

                self.S_t = (
                    -(self.v_hat + self.omega * self.c_t)).argsort()[
                    0:self.scenario.number_cores
                ]

            except Exception as e:
                print(e)
                self.S_t = (
                    -(self.v_hat)
                ).argsort()[0:self.scenario.number_cores]
            else:
                self.S_t = (
                    -(self.v_hat)
                ).argsort()[0:self.scenario.number_cores]

            if self.scenario.verbosity in (1, 2) and len(self.S_t_prior) != 0:
                print('\nSkill and confidence assessment of the contenders',
                      'from tournament:\n \nContender',
                      ' ' * 34, '   ( v_hat', ' ' * 12, ', c_t', + 14 * ' ',
                      ')')
                pool_ids = list(self.pool.keys())
                for core in range(self.scenario.number_cores):
                    print(pool_ids[self.S_t_prior[core]], 'assessment is:',
                          '(', self.v_hat[self.S_t_prior[core]], ',',
                          self.c_t[self.S_t_prior[core]], ')')
                print('\n\n')

    def assess_children(
            self, transformed_configs: np.ndarray, nd: int) -> list[int]:
        """
        Compute the skill and confidence values for newly generated 
        Configurations based on feature matrix computed for the set of newly 
        generated configurations and return the indices of nd best 
        of them.

        Parameters
        ----------
        transformed_configs : np.ndarray
            transformed values of generated Configurations
        nd : int
            Number of configurations to be replaced in the pool.

        Returns
        -------
        list[int]
            List of indices of the nd best Configurations from all newly 
            generated Configurations.
        """
        with threadpool_limits(limits=1):
            # self.t = self.scenario.cpplt
            ltc = len(transformed_configs)
            X_t = np.zeros((ltc, len(self.theta_bar)), dtype=np.float32)
            for i in range(len(transformed_configs)):
                X_t[i, :] = self.joinFeatureMap(
                    transformed_configs[i,], 
                    self.scaled_features[0],
                    self.scenario.jfm
                )

            normalize(X_t, norm='max', copy=False)

            # Estimated skill parameters
            v_hat = np.zeros(ltc).astype('float64')
            for i in range(ltc):
                v_hat[i] = np.exp(np.inner(self.theta_bar, X_t[i, :]))

            # Confidences
            c_t = np.zeros(ltc).astype('float64')

            # Configuration quality estimation
            try:
                V_hat = (1 / self.t) * self.grad_op_sum
                V_hat = V_hat.astype('float64')
                S_hat = (1 / self.t) * self.hess_sum
                S_hat = S_hat.astype('float64')
                try:
                    S_hat_inv = np.linalg.inv(S_hat)
                except Exception as e:
                    print(e)
                    S_hat_inv = np.linalg.pinv(S_hat)

                S_hat_inv = S_hat_inv.astype('float64')
                Sigma_hat = (1 / self.t) * np.dot(
                    np.dot(S_hat_inv, V_hat),
                    S_hat_inv)
                Sigma_hat_sqrt = sp.linalg.sqrtm(Sigma_hat)

                for i in range(ltc):
                    M_i = np.exp(2 * np.dot(
                        X_t[i], self.theta_bar.copy())
                    ) * np.outer(X_t[i], X_t[i])

                    c_t[i] = np.sqrt((
                        2 * np.log(self.t) + self.len_theta_bar + 2 * np.sqrt(
                            self.len_theta_bar * np.log(self.t))
                    ) * np.linalg.norm(
                        np.dot(np.dot(Sigma_hat_sqrt, M_i), Sigma_hat_sqrt)))

                if np.any(np.isinf(c_t)):
                    self.c_t = np.ones_like(c_t)

                elif max(c_t) != 0:
                    mc = max(c_t)
                    for i, c in enumerate(c_t):
                        if c != 0:
                            c_t[i] = c_t[i] / mc

                v_hat = v_hat / max(v_hat)

                S_t = (
                    -(v_hat + self.omega * c_t)).argsort()

            except Exception as e:
                print(e)
                S_t = (-(v_hat)).argsort()
                pass

            self.t = self.scenario.cpplt

        return S_t

    def asses_configurations(self) -> None:
        """
        Comparatively asses which Configurations are to be replaced from the 
        pool based on skills and confidence computed by CPPL model.

        Returns
        -------
        None
        """
        self.discard = []

        for i in range(self.pool_size):
            for j in range(self.pool_size):
                if j != i and \
                    self.v_hat[j] - self.scenario.kappa * self.c_t[j] \
                        > self.v_hat[i] + self.scenario.kappa * self.c_t[i] \
                        and (not (i in self.discard)):
                    self.discard.append(i)
                    break

    def select_contenders(self) -> None:
        """
        Select Configurations to run in next tournament and save them in 
        self.contender_dict.

        Returns
        -------
        None
        """
        self.contender_dict = {}
        pool_list = list(copy.deepcopy(self.pool).values())
        best = []
        for i in range(self.keeptop):
            best.append(self.S_t[i])

        temp_pool = copy.copy(self.pool)
        for b in best:
            del temp_pool[pool_list[b].id]
        random_pick = \
            random.sample(
                list(temp_pool.keys()),
                self.scenario.number_cores - self.keeptop)

        if self.scenario.epsilon_greedy:
            for s in self.S_t[:self.keeptop]:
                self.contender_dict[pool_list[s].id] = pool_list[s]
        else:
            for s in self.S_t:
                self.contender_dict[pool_list[s].id] = pool_list[s]

        rest_s = copy.deepcopy(self.S_t[self.keeptop:])

        if self.scenario.epsilon_greedy:
            for i, s in enumerate(rest_s):
                if random.random() < self.scenario.epsilon and \
                        s not in best and \
                        random_pick[i] not in self.contender_dict:
                    self.contender_dict[random_pick[i]] = \
                        temp_pool[random_pick[i]]
                elif pool_list[s].id not in self.contender_dict:
                    self.contender_dict[pool_list[s].id] = pool_list[s]
                else:
                    rn = random.randint(0, self.pool_size - 1)
                    while pool_list[rn].id in self.contender_dict:
                        rn = random.randint(0, self.pool_size - 1)
                    self.contender_dict[pool_list[rn].id] = pool_list[rn]

        self.prior_contender_dict = self.contender_dict

    def init_param_scaler(self, config_space: argparse.Namespace) -> None:
        """
        Initialize MinMaxScaler to be used on Configuration values.

        Parameters
        ----------
        config_space : argparse.Namespace
            Namespace containing all settings for the RTAC.

        Returns
        -------
        None
        """
        self.split_param_types(config_space)
        self.min_max_scaler = MinMaxScaler()
        self.min_max_scaler.fit([self.lower_bounds, self.upper_bounds])

    def split_param_types(self, config_space: argparse.Namespace) -> None:
        """
        Split parameter space by categorical and everything else.

        Parameters
        ----------
        config_space : argparse.Namespace
            Namespace containing all settings for the RTAC.

        Returns
        -------
        None
        """

        # Split categorical parameters and rest for later OneHotEncoding
        self.non_cat_param_names = []
        self.cat_param_names = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.cat_values = []

        if not isinstance(config_space, ConfigurationSpace):
            for param, definition in config_space.items():
                if definition.paramtype is ParamType.categorical:
                    if definition.valtype == ValType.str:
                        self.cat_param_names.append(param)
                        self.cat_values.append(definition.values)
                    elif definition.valtype == ValType.int:
                        self.cat_param_names.append(param)
                        self.cat_values.append(
                            [i for i in range(
                                definition.minval, definition.maxval + 1)])
                else:
                    self.non_cat_param_names.append(param)
                    self.lower_bounds.append(definition.minval)
                    self.upper_bounds.append(definition.maxval)
                    
        else:
            for param_name in config_space:
                param = config_space.get_hyperparameter(param_name)
                if isinstance(param, CategoricalHyperparameter):
                    self.cat_param_names.append(param_name) 
                    self.cat_values.append(param.choices)
                else:
                    self.non_cat_param_names.append(param_name)
                    self.lower_bounds.append(param.lower)
                    self.upper_bounds.append(param.upper)

    def init_onehot_encoder(self) -> None:
        """
        Initialize OneHotEncoder to be used on categorical Configuration 
        values.

        Returns
        -------
        None
        """
        self.one_hot_enc = \
            OneHotEncoder(categories=self.cat_values,
                          sparse_output=False,
                          handle_unknown='ignore')

    def transform_conf(self, conf: Configuration) -> np.ndarray:
        """
        Transorm Configuration values with MinMaxScaler and OneHotEncoder.

        Parameters
        ----------
        conf : Configuration
            Configuration to be transformed.

        Returns
        -------
        np.ndarray
            Transformed Configuration values.
        """
        with threadpool_limits(limits=1):
            ncvt = self.min_max_scaler.transform([[conf.conf[k]
                                                 for k in
                                                 self.non_cat_param_names]])
            cvt = self.one_hot_enc.fit_transform([[conf.conf[k]
                                                 for k in
                                                 self.cat_param_names]])
            # One-hot encoding increases impact of the feature in the model
            # so we reduce the impact by the number of binary features created
            normalized = []
            start = 0
            for cat in self.cat_values:
                num_cats = len(cat)
                segment = cvt[0, start:start + num_cats]
                normalized.extend(segment / num_cats)
                start += num_cats

            cvt = np.array(normalized).reshape(1, -1)

        return np.concatenate((ncvt, cvt), axis=None)

    def pre_train(self, feature_path: str) -> None:
        """
        Pretrain StandardScaler to be used on problem instance features.

        Parameters
        ----------
        feature_path : str
            Path to the CSV file containing the features.

        Returns
        -------
        None
        """
        import ast
        features = []
        with open(f'{feature_path}') as f:
            for line in f:
                d = ast.literal_eval(line)
                array = np.array([v if v is not None else 0 for v in d])
                features.append(array)

        self.features = np.array(features)
        self.standard_scaler.fit(self.features)
        self.pca_obj_inst.fit(self.standard_scaler.transform(self.features))
