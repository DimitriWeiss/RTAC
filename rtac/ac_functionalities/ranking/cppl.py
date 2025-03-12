import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures
)
import importlib
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


class CPPL():
    """CPPL Bandit and AC functions."""

    def __init__(self, scenario, pool):
        """Initialize CPPL object."""
        self.scenario = scenario
        self.pool = pool
        self.transformed_pool = {}
        self.features = []
        self.get_param_types()
        self.standard_scaler = StandardScaler()
        
        fg_module = importlib.import_module(self.scenario.feature_gen)
        fg_name = self.scenario.feature_gen_name
        self.feature_gen = getattr(fg_module, fg_name)()

        # Calibrate MinMaxScaler for non-categorical parameters
        self.min_max_scaler = MinMaxScaler()
        lower_bounds, upper_bounds = self.get_scaling_bounds()
        lower_bounds_nc, _ = self.get_param_types(lower_bounds)
        upper_bounds_nc, _ = self.get_param_types(upper_bounds)
        params = np.append([lower_bounds_nc], [upper_bounds_nc], axis=0)
        self.min_max_scaler.fit(params)

        self.init_onehot_encoder()
        for conf in self.pool:
            self.transformed_pool[conf.id] = self.transform_conf(conf)

        self.pca_obj_inst = PCA(n_components=self.scenario.nc_pca_f)
        self.pca_obj_params = PCA(n_components=self.scenario.nc_pca_p)
         
        if self.scenario.jfm == 'concatenation':
            d = self.scenario.nc_pca_f + self.scenario.nc_pca_p
        elif self.scenario.jfm == 'kronecker':
            d = self.scenario.nc_pca_f * self.scenario.nc_pca_p
        elif self.scenario.jfm == 'polynomial':
            d = 4
            for i in range((
                    self.scenario.nc_pca_f + self.scenario.nc_pca_p) - 2):
                d = d + 3 + i

        self.theta_hat = np.zeros(d)
        self.theta_bar = self.theta_hat

        self.grad_op_sum = np.zeros((d, d))
        self.hess_sum = np.zeros((d, d))
        self.omega = self.scenario.omega
        self.gamma_1 = self.scenario.gamma
        self.alpha = self.scenario.alpha
        self.t = 0
        self.Y_t = 0
        self.S_t = []
        self.grad = np.zeros(d)

    def process_results(self, rtac_data):
        self.grad = self.gradient(
            self.theta_hat, self.rtac_data.winner.value, self.S_t, self.X_t)
        self.theta_hat = \
            self.theta_hat + self.gamma_1 * self.t ** (-self.alpha) * self.grad

        # Truncating
        # theta_hat[theta_hat < 0] = 0
        # theta_hat[theta_hat > 1] = 1

        # Max norming
        # self.theta_hat = self.theta_hat / max(self.theta_hat)
        # 0-1 Norming
        # self.theta_hat = \
        #    (self.theta_hat - min(self.theta_hat)) / \
        #    (max(self.theta_hat) - min(self.theta_hat))
        
        # Update theta_bar
        self.theta_bar = \
            (self.t - 1) * self.theta_bar / self.t + self.theta_hat / self.t

    def gradient(self):
        n = np.zeros((len(self.theta)))
        d = 0
        for i in self.S_t:
            e_theta_X = np.exp(np.dot(self.theta, self.X_t[i, :]))
            n = n + (self.X_t[i, :] * e_theta_X)
            d = d + e_theta_X

        self.n = n
        self.d = d

        return self.X_t[self.Y_t, :] - (n / d)

    def hessian(self):
        n_2 = 0
        for i in self.S_t:
            e_theta_X = np.exp(np.dot(self.theta, self.X_t[i, :]))
            n_2 = n_2 + (e_theta_X * np.outer(self.X_t[i, :], self.X_t[i, :]))

        return (np.outer(self.n, self.n) / (self.d ** 2)) - (n_2 / self.d)

    def joinFeatureMap(x, y, mode):
        if mode == 'concatenation':
            return np.concatenate((x, y), axis=0) 
        elif mode == 'kronecker':
            return np.kron(x, y)
        elif mode == 'polynomial':
            poly = PolynomialFeatures(degree=2, interaction_only=True)
            return poly.fit_transform(
                np.concatenate((x, y), axis=0).reshape(1, -1))

    def manage_pool(self):
        n_discarded = self.discard_configs()
        self.generate_configs(n_discarded)

    def select_contenders(self, instance):
        new_features = self.feature_gen.get_features(instance)
        self.standard_scaler.partial_fit(new_features)
        self.features = np.asarray(self.features.append(new_features))
        if len(self.features) > 100:
            self.features = self.features[len(self.features) - 1:]
        scaled_features = self.standard_scaler.transform(new_features)
        # scaled_features = self.standard_scaler.fit_transform(self.features)

        # PCA on features
        features_transformed = self.pca_obj_inst.fit_transform(scaled_features)

        params, _ = read_parametrizations(Pool,solver,target_directory)

        params = np.asarray(params)

        params_original_size = params.shape[1]

        #params = log_on_huge_params.log_space_convert(solver,pl,params,json_param_file)

        params = self.min_max_scaler.transform(params)

        # PCA on parametrizations
        params_transformed = self.pca_obj_params.transform(params)

    def get_scaling_bounds(self):
        lower_bounds = []
        upper_bounds = []
        for param, definition in self.config_space.items():
            # Leavin out categorical parameters which will be OneHot encoded
            if definition.paramtype in \
                    (ParamType.discrete, ParamType.continuous):
                lower_bounds.append(definition.minval)
                upper_bounds.append(definition.maxval)

        return lower_bounds, upper_bounds

    def get_param_types(self):
        self.non_cat_param_names = []
        self.cat_param_names = []
        for param, definition in self.config_space.items():
            # Leavin out categorical parameters which will be OneHot encoded
            if definition.paramtype in \
                    (ParamType.discrete, ParamType.continuous):
                self.non_cat_param_names.append(param)
            else:
                self.cat_param_names.append(param)

    def split_param_types(self, config):
        non_cat_params = [config[param] for param in self.non_cat_param_names]
        cat_params = [config[param] for param in self.cat_param_names]

        return non_cat_params, cat_params

    def init_onehot_encoder(self):
        self.o_h_enc = OneHotEncoder(categories=list_of_bounds)
        self.o_h_enc.fit(np.array(list_o_of_default).reshape(1, -1))

    def one_hot_encode(self, config):
        pass

    def transform_conf(self, conf):
        non_cat_params, cat_params = self.split_param_types(conf)
        non_cat_params = self.min_max_scaler.transform(non_cat_params)
        cat_params = self.one_hot_encode(cat_params)

        return non_cat_params + cat_params
