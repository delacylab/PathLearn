########################################################################################################################
# Apache License 2.0
########################################################################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2024 Nina de Lacy

########################################################################################################################
# Disclaimer and References
########################################################################################################################
# A large portion of this script is borrowed from BorutaPy (by Daniel Homola <dani.homola@gmail.com>), a Python
# package of the Boruta feature selection method.
# (1) For a theoretical discussion of the Boruta method, see Kursa, M.B., Jankowski, A., & Rudnicki, W.R. (2010).
# Boruta - A System for Feature Selection. Fundam. Informaticae, 101, 271-285.
# https://content.iospress.com/articles/fundamenta-informaticae/fi101-4-02
# (2) For the R implementation of Boruta, see Kursa, M. B., & Rudnicki, W. R. (2010). Feature Selection with the Boruta
# Package. Journal of Statistical Software, 36(11), 1–13. https://doi.org/10.18637/jss.v036.i11
# (3) For the BorutaPy Python implementation, see https://github.com/scikit-learn-contrib/boruta_py.

########################################################################################################################
# Remark: Modifications to BorutaPy
########################################################################################################################
# Compared to the BorutaPy package, the main modification made in this script is how runs under various hyperparamter
# configurations can be done simultaneously without executing independent runs. This optimization technique is motivated
# by the fact that different feature subsets returned by different selection criteria may be needed for subsequent
# prediction tasks to improve performance.
#
# As suggested in the BorutaPy documentation, using the maximum importance scores of the shadow features (shadow
# importances) "could be overly harsh". Thus, they proposed tuning the percentile (perc) of the shadow importances
# to control for the strictness of comparison. However, identifying a "good" perc can be time-inefficient when the list
# of choices is long (e.g., 75, 76, ..., 99, 100), not to mention the combinatorial complication when the user is
# interested to fine-tune the level of significance (alpha) used in embedded statistical tests as well.
#
# Our optimization technique relies on a voting mechanism across different hyperparameter configurations to decide which
# feature to reject/exclude in subsequent iterations. Users can fine-tune two hyperparameters: perc and alpha. Each can
# be specified as a list of values rather than a single value in BorutaPy. Notice that once a feature is excluded in a
# given iteration, it will not be considered in any subsequent iteration in the random forest modeling process. However,
# different hyperparamter configurations can disagree on which feature to reject. To address this, we propose three
# different voting rules (vote_rule) to resolve the disagreement.

# [vote_rule=0, unanimous rule] A feature is excluded in subsequent iterations if ALL hyperparamter configurations
# reject the feature.
# [vote_rule=1, majority rule] A feature is excluded in subsequent iterations if AT LEAST HALF hyperparamter
# configurations reject the feature.
# [vote_rule=2, minority rule] A feature is excluded in subsequent iterations if AT LEAST ONE hyperparamter
# configuration rejects the feature.

# The minority rule represents the most time-efficient optimization but bears the highest risk of Type-2 errors. In
# addition, another modification made in this script is the disregard of "tentative" features (i.e., features neither
# accepted nor rejected as relevant after all iterations are run). A crucial merit of considering tentative features is
# to minimize Type-II errors. However, it is not clear how this tentativeness is related to feature relevance in
# principle. Also, our optimization attains the same goal by motivating users to run Boruta under various hyperparameter
# configurations for more robust results in subsequent prediction processes. For users who are interested in the
# tentative feature subset returned under a specific hyperparameter configuration, they are suggested to use the original
# BorutaPy implementation.

########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
import scipy as sp
from itertools import product
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import _is_fitted
from statsmodels.stats.multitest import fdrcorrection
from typing import Literal, Optional, Union

########################################################################################################################
# Define a Boruta model as a class
########################################################################################################################


class BorutaClass(BaseEstimator, TransformerMixin):
    """
    A. Runtime parameters
    ---------------------
    A1. estimator: A RandomForestClassifier/RandomForestRegressor object from sklearn.ensemble.
        See the original documentation for its runtime parameter setting.
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    A2. n_estimators: A positive integer or 'auto'.
        The number of estimators in the chosen ensemble method in A1. If 'auto', the number of estimators is
        determined automatically based on the size of the feature set.
        Default setting: n_estimators='auto'
    A3. perc: A (list of) positive integer(s) in the close interval [1, 100].
        The percentile of the shadow importance to be compared to for each true feature in each iteration.
        Default setting: perc=100
    A4. alpha: A (list of) float(s) in the open interval (0, 1).
        The level of significance used to compare with the p-values in each two-sided statistical hypothesis test.
        Default setting: alpha=0.05
    A5. two_step: A boolean.
        Using Bonferroni correction for p-values if True or Benjamini-Hochberg FDR then Bonferroni correction otherwise.
        Default setting: two_step=True
    A6. max_iter: A positive integer.
        Number of maximum iterations to run.
        Default setting: max_iter=100
    A7. vote_rule: An integer in [0, 1, 2].
        Different voting rules to resolve disagreement of rejection across different hyperparameter configurations
        specified in A3 and A4 for each iteration. See "Remark: Modifications to BorutaPy" for details.
        Default setting: vote_rule=0
    A8. random_state: An integer or a numpy RNG object.
        Random seed used by the random number generator.
        Default setting: random_state=None
    A9. verbose: An integer in [0, 1, 2].
        Verbosity. No logging if 0, displaying iteration numbers and number of trees in the Random Forest model used
        if 1, and together with the indices of the selected feature subset for each hyperparameter configuration if 2.
        Default setting: verbose=1

    B. Attributes
    -------------
    B1. importance_history: A two-dimensional numpy array.
        A matrix storing the importance scores of the real features for each iteration. It has a dimension of
        (number of iterations executed, number of real features).
    B2. hit_dict: A dictionary.
        A dictionary with keys as the hyperparameter configurations and values as the number of hits for each feature
        over all iterations.
    B3. decision_dict: A dictionary.
        A dictionary with keys as the hyperparameter configurations and values as the list of decisions made to each
        feature (encoded by 1 as accepted, -1 as rejected, and 0 as neither).
    (A1-A9 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. get_shuffle(input_array)
        Shuffle the input numpy array.
        :param: input_array: A two-dimensional numpy array.
        :return: A shuffled input_array.
    C2. get_shadow(input_array)
        Obtain the shadow copy of the input numpy array.
        :param: input_array: A two-dimensional numpy array.
        :return: A two-dimensional numpy array, encoding the shadow copies of the true features in the input_array.
    C3. get_importance(X, y)
        Fit the estimator in A1 with (feature set X, target y), then return the importance scores of each feature.
        :param: X: A two-dimensional numpy array. Feature set.
        :param: y: A one-dimensional numpy array. Target.
        :return: A list of importance scores, one for each feature in X.
    C4. get_model()
        Get the fitted model.
        :return: The (fitted) estimator in the last iteration.
    C5. get_tree_num(n_feat)
        Compute number of trees needed from the number of features.
        :param: n_feat: An integer. The number of features.
        :return: An integer as the optimal number of trees (i.e., n_estimators).
    C6. get_rejection()
        Identify the (un-)rejected features.
        :return:
        (a) The indices of the rejected features.
        (b) The indices of the un-rejected features.
    C7. do_tests(n_iter)
        Perform statistical tests to accept/reject relative to a given number of iterations performed (n_iter).
        :param: A given number of iterations.
    C8. fit(X, y)
        Fit the estimator in A1 with (feature set X, target y).
        :param X: A two-dimensional numpy array. Samples of the feature set with dimension as (n_samples, n_features).
        :param y: A one-dimensional numpy array. Samples of the target dimension as (n_samples).
    C9. get_rlv_feat()
        Identify the relevant features.
        :return: A dictionary with keys as the hyperparameter configurations and values as lists of indices encoding
        the feature subset selected.

    References
    ----------
    1. BorutaPy Python package. https://github.com/scikit-learn-contrib/boruta_py.
    2. Kursa, M. B., & Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. Journal of Statistical
       Software, 36(11), 1–13. https://doi.org/10.18637/jss.v036.i11.
    """
    def __init__(self,
                 estimator: Union[RandomForestClassifier, RandomForestRegressor],
                 n_estimators: Union[Literal['auto'], int] = 'auto',
                 perc: Union[int, list[int]] = 100,
                 alpha: Union[float, list[float]] = 0.05,
                 two_step: bool = True,
                 max_iter: int = 100,
                 vote_rule: Literal[0, 1, 2] = 0,
                 random_state: Optional[Union[int, np.random.Generator]] = None,
                 verbose: Literal[0, 1, 2] = 1):

        # Type and value check
        assert type(estimator) in [RandomForestClassifier, RandomForestRegressor], \
            (f"RF_estimator must be a RandomForestRegressor or RandomForestClassifier object from sklearn.ensemble. "
             f"Now its type is {type(estimator)}.")
        self.estimator = estimator
        assert (isinstance(n_estimators, int) and n_estimators >= 1) or n_estimators == 'auto', \
            f"n_estimators (in Boruta) must be a positive integer or equal to 'auto'. Now its value is {n_estimators}."
        self.n_estimators = n_estimators
        if isinstance(perc, int):
            assert 1 <= perc <= 100, \
                f"perc (in Boruta), if integer, must be in the close interval [1, 100]."
        elif isinstance(perc, list):
            assert all([isinstance(p, int) and (1 <= p <= 100) for p in perc]), \
                f"perc (in Boruta), if list, must be a list of positive integers in the close interval [1, 100]."
        else:
            raise TypeError(f'perc (in Boruta) must be an integer or a list or integers. Now its type is {type(perc)}.')
        self.perc = [perc] if isinstance(perc, int) else perc
        if isinstance(alpha, float):
            assert 0 < alpha < 1, \
                f"alpha (in Boruta), if float, must be in the open interval (0, 1)"
        elif isinstance(alpha, list):
                assert all([isinstance(a, float) and (0 < a < 1) for a in alpha]), \
                    f'alpha (in Boruta), if list, must be a list of floats in the open interval (0, 1).'
        else:
            raise TypeError(f'alpha (in Boruta) must be a float or a list of floats. Now its type is {type(alpha)}.')
        self.alpha = [alpha] if isinstance(alpha, float) else alpha
        assert isinstance(two_step, bool), \
            f"two_step (in Boruta) must be a boolean. Now its value is {two_step}."
        self.two_step = two_step
        assert isinstance(max_iter, int) and max_iter > 0, \
            f"max_iter (in Boruta) must be a positive integer. Now its values is {max_iter}."
        self.max_iter = max_iter
        assert vote_rule in [0, 1, 2], \
            f"vote_rule (in Boruta) must be in [0, 1, 2]. Now its value is {vote_rule}"
        self.vote_rule = vote_rule
        self.random_state = check_random_state(random_state)      # Rely on sklearn to check if random_state is valid
        assert verbose in [0, 1, 2], \
            f"verbose (in Boruta) must be in [0, 1, 2]. Now its value is {verbose}."
        self.verbose = verbose

        # Specify all the hyperparameter configurations
        self.importance_history = None
        self.hit_dict = {}
        self.decision_dict = {}
        for hp_config in product(self.perc, self.alpha):      # hyperparameter configurations
            self.hit_dict[hp_config], self.decision_dict[hp_config] = None, None

    def get_shuffle(self, input_array: np.ndarray):
        self.random_state.shuffle(input_array)    # Rely on numpy to check if input_array is valid
        return input_array

    def get_shadow(self, input_array: np.ndarray):
        X_shadow = np.copy(input_array)     # Rely on numpy to check if input_array is valid
        # Guarantee that we always have at least 5 shadow features for importance comparison
        while X_shadow.shape[1] < 5:
            X_shadow = np.hstack((X_shadow, X_shadow))
        X_shadow = np.apply_along_axis(self.get_shuffle, 0, X_shadow)
        return X_shadow

    def get_importance(self, X: np.ndarray, y: np.ndarray):
        try:
            self.estimator.fit(X, y)    # Rely on sklearn to check if X and y are valid
        except Exception as e:
            raise ValueError("Please check your X and y. The provided estimator cannot be fitted to your data."
                             "\n" + str(e))
        try:
            imp = self.estimator.feature_importances_
        except Exception:
            raise ValueError("Only methods with .feature_importance_ attribute are currently supported.")
        return imp

    def _get_model(self):
        # Return the estimator (usually after model fitting) in case the user needs it.
        return self.estimator

    def get_tree_num(self, n_feat: int):
        assert isinstance(n_feat, int), \
            f'n_feat must be an integer. Now its type is {type(n_feat)}.'
        assert n_feat >= 1, f'n_feat must be a positive integer. Now its value is {n_feat}.'
        depth = self.estimator.get_params()['max_depth']
        if depth is None:
            depth = 7
        f_repr = 100            # how many times a feature should be considered on average
        multi = ((n_feat * 2) / (np.sqrt(n_feat * 2) * depth))
        # n_feat * 2 because the training matrix is extended with n shadow features
        n_estimators = int(multi * f_repr)
        return n_estimators

    def get_rejection(self):
        first_dec_reg = list(self.decision_dict.values())[0]
        assert first_dec_reg is not None
        full_features = set(range(len(first_dec_reg)))
        if self.vote_rule == 0:       # unanimous rule
            not_rejected = set()
            for dec_reg in self.decision_dict.values():
                not_rejected |= set(np.where(dec_reg >= 0)[0])
            rejected = full_features.difference(not_rejected)
        elif self.vote_rule == 1:     # majority rule
            n_hp_config = len(self.decision_dict.keys())
            rejected_dict = {feat: 0 for feat in full_features}
            for dec_reg in self.decision_dict.values():
                rejected_cur = np.where(dec_reg == -1)[0]
                for feat in rejected_cur:
                    rejected_dict[feat] += 1
            rejected = set(feat for feat, count in rejected_dict.items() if count / n_hp_config >= 0.5)
            not_rejected = full_features.difference(rejected)
        else:                       # minority rule
            rejected = set()
            for dec_reg in self.decision_dict.values():
                rejected |= set(np.where(dec_reg == -1)[0])
            not_rejected = full_features.difference(rejected)
        return sorted(rejected), sorted(not_rejected)

    def do_tests(self, n_iter: int):
        assert isinstance(n_iter, int), f'n_iter must be an integer. Now its type is {type(n_iter)}.'
        assert n_iter >= 0, f'n_iter must be a non-negative integer. Now its value is {n_iter}.'
        for hp_config in self.decision_dict.keys():
            perc_value, alpha_value = hp_config[0], hp_config[1]
            dec_reg = self.decision_dict[hp_config]
            hit_reg = self.hit_dict[hp_config]
            active_features = np.where(dec_reg >= 0)[0]
            hits = hit_reg[active_features]

            # get uncorrected p-values based on hit_reg
            to_accept_ps = sp.stats.binom.sf(hits-1, n_iter, .5).flatten()
            to_reject_ps = sp.stats.binom.cdf(hits, n_iter, .5).flatten()

            if self.two_step:
                to_accept = fdrcorrection(to_accept_ps)[1]
                to_reject = fdrcorrection(to_reject_ps)[1]
                to_accept2 = to_accept_ps <= alpha_value / float(n_iter)
                to_reject2 = to_reject_ps <= alpha_value / float(n_iter)
                to_accept *= to_accept2
                to_reject *= to_reject2
            else:
                to_accept = to_accept_ps <= alpha_value / float(len(dec_reg))
                to_reject = to_reject_ps <= alpha_value / float(len(dec_reg))

            to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
            to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

            # update dec_reg
            dec_reg[active_features[to_accept]] = 1
            dec_reg[active_features[to_reject]] = -1
            self.decision_dict[hp_config] = dec_reg

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            X = np.array(X)
        except:
            raise TypeError(f'X must be (convertible to) a numpy array. Now its type is {type(X)}.')
        try:
            y = np.array(y)
        except:
            raise TypeError(f'y must be (convertible to) a numpy.array. Now its type is {type(y)}.')
        assert len(X.shape) == 2, \
            f'X must be two-dimensional. Now its dimension is {X.shape}'
        assert len(y.shape) == 1, \
            f'y must be one-dimensional. Now its dimension is {y.shape}.'
        n_sample, n_feat = X.shape
        _iter = 1
        for hp_config in product(self.perc, self.alpha):
            if self.hit_dict[hp_config] is None:
                self.hit_dict[hp_config] = np.zeros(n_feat, dtype=np.int32)
            if self.decision_dict[hp_config] is None:
                self.decision_dict[hp_config] = np.zeros(n_feat, dtype=np.int32)
        if self.importance_history is None:
            self.importance_history = np.zeros(n_feat, dtype=np.int32)

        # set n_estimators if not already specified
        if self.n_estimators != 'auto':
            self.estimator.set_params(n_estimators=self.n_estimators)

        # main feature selection loop
        while _iter <= self.max_iter and any([np.any(v == 0) for v in self.decision_dict.values()]):
            # the second condition above ensures that the loop will continue unless all hp_config have no features
            # left un-rejected or un-accepted.
            rejected, unrejected = self.get_rejection()
            n_unrejected = len(unrejected)
            if self.n_estimators == 'auto':
                # find optimal number of trees and depth
                n_tree = self.get_tree_num(n_unrejected)
                self.estimator.set_params(n_estimators=n_tree)
            else:
                n_tree = self.n_estimators

            # Obtain shadow features
            X_unrejected = X[:, unrejected]
            X_shadow = self.get_shadow(X_unrejected)
            if self.verbose in [1, 2]:
                print(f"Iteration {_iter}: Fitting a Random Forest model (of {n_tree} trees) with {n_unrejected} real "
                      f"features and {X_shadow.shape[1]} shadow features.", flush=True)

            # Fit model and get importances
            importance = self.get_importance(np.hstack((X_unrejected, X_shadow)), y)
            importance_shadow = importance[n_unrejected:]
            importance_real = np.full(X.shape[1], np.nan)
            importance_real[unrejected] = importance[:n_unrejected]
            self.importance_history = np.vstack((self.importance_history, importance_real))
            importance_real_no_nan = importance_real.copy()
            importance_real_no_nan[np.isnan(importance_real_no_nan)] = 0

            # Register hits according to different perc values
            for perc_value in self.perc:
                shadow_importance_target = np.percentile(importance_shadow, perc_value)
                hits = np.where(importance_real_no_nan > shadow_importance_target)[0]
                for k in self.hit_dict.keys():
                    if k[0] == perc_value:
                        self.hit_dict[k][hits] += 1

            self.do_tests(_iter)
            _iter += 1
            if self.verbose == 2:
                for config_idx, hp_config in enumerate(self.decision_dict.keys(), 1):
                    perc_value, alpha_value = hp_config[0], hp_config[1]
                    dec_reg = self.decision_dict[hp_config]
                    empty_line = "\n" if config_idx == len(self.decision_dict.keys()) else ""
                    print(f"Configuration {config_idx}; (perc={perc_value}, alpha={alpha_value}). "
                          f"Accepted indices: {list(np.where(dec_reg == 1)[0])}{empty_line}", flush=True)

    def get_rlv_feat(self):
        # Return
        if not _is_fitted(self.estimator):
            raise ValueError('The Boruta model has not been fitted.')
        support_dict = {}
        for config_idx, ((perc_value, alpha_value), v) in enumerate(self.decision_dict.items(), 1):
            support = list(np.where(np.array(v) == 1)[0])
            support_dict[(perc_value, alpha_value)] = support
        return support_dict

########################################################################################################################
