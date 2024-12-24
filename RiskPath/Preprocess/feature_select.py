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
# Import packages
########################################################################################################################
import numpy as np
from .boruta_optimized import BorutaClass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.utils.validation import _is_fitted
from typing import Literal, Optional, Union

########################################################################################################################
# Define feature selection class for classification
########################################################################################################################


class FSClassifier(BaseEstimator, TransformerMixin):
    """
    A. Runtime parameters
    ---------------------
    A1. L_estimator: A LogisticRegressionCV object from sklearn.linear_model.
        Estimate the feature subset linearly relevant to the target variable. See the original documentation for its
        runtime parameter setting.
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html.
        Default setting: L_estimator=LogisticRegressionCV(Cs=100, cv=5, penalty='l1', solver='saga', max_iter=1000,
                         n_jobs=-1, random_state=42)
    A2. L_beta: A non-negative float.
        Strictness of the linear feature selection process where larger values indicate a stricter criterion.
        Default setting: L_beta=0
    A3. RF_estimator: A RandomForestRegressor object from sklearn.ensemble.
        The embedded model in Boruta that aims to return the feature subset non-linearly relevant to the target variable.
        See the original documentation for its runtime parameter setting.
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        Default setting: RF_estimator=RandomForestClassifier(max_depth=7, n_jobs=-1)
    A4. B_n_estimators: a positive integer or 'auto'.
        The number of estimators in the chosen ensemble method in A3. If 'auto', the number of estimators is
        determined automatically based on the size of the feature set.
        Default setting: B_n_estimators='auto'
    A5. B_perc: A (list of) positive integer(s) in the close interval [1, 100].
        The percentile of the shadow importance in Boruta to be compared to for each true feature in each iteration.
        Default setting: B_perc=100
    A6. B_alpha: A (list of) float(s) in the open interval (0, 1).
        The level of significance in Boruta used to compare with the p-values in each two-sided statistical hypothesis
        test.
        Default setting: B_alpha=0.05
    A7. B_two_step: A boolean.
        Using Bonferroni correction for p-values if True or Benjamini-Hochberg FDR then Bonferroni correction otherwise
        in Boruta.
        Default setting: B_two_step=True
    A8. B_max_iter: A positive integer.
        Number of maximum iterations to run in Boruta.
        Default setting: B_max_iter=100
    A9. B_vote_rule: An integer in [0, 1, 2].
        Different voting rules in Boruta to resolve disagreement of rejection across different hyperparameter
        configurations specified in A3 and A4 for each iteration. See "Remark: Modifications to BorutaPy" in
        boruta_optimized.py for details.
        Default setting: B_vote_rule=0
    A10. B_random_state: An integer or a numpy RNG object.
         Random seed used by the random number generator in Boruta.
         Default setting: B_random_state=None
    A11. B_verbose: An integer in [0, 1, 2].
         Verbosity in Boruta. No logging if 0, displaying iteration numbers and number of trees in the Random Forest
         model used if 1, and together with the indices of the selected feature subset for each hyperparameter
         configuration if 2.
         Default setting: B_verbose=1

    B. Attributes
    -------------
    B1. B_model: A BorutaClass object with runtime parameters specified in A3-A11
    (A1-A3 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. fit_linear(X, y)
        Fit the LogisticRegressionCV model with (feature set X, target y) where
        :param X: A two-dimensional numpy array. Samples of the feature set with dimension as (n_samples, n_features).
        :param y: A one-dimensional numpy array. Samples of the target with dimension as (n_samples).
    C2. fit_nonlinear(X, y)
        Fit the Boruta model with (feature set X, target y) where
        :param X: A two-dimensional numpy array. Samples of the feature set with dimension as (n_samples, n_features).
        :param y: A one-dimensional numpy array. Samples of the target with dimension as (n_samples).
    C3. fit_both(X, y)
        Fit both LogisticRegressionCV and Boruta models with (feature set X, target y) where
        :param X: A two-dimensional numpy array. Samples of the feature set with dimension as (n_samples, n_features).
        :param y: A one-dimensional numpy array. Samples of the target with dimension as (n_samples).
    C4. get_linear_coef()
        :return: A list of linear coefficients in the fitted LogisticRegressionCV model.
    C5. get_linear_rlv_feat()
        :return: A list of indices of the relevant feature subset selected by the LogisticRegressionCV model.
    C6. get_nonlinear_rlv_feat()
        :return: A list of indices of the relevant feature subset selected by the Boruta model if only one
        hyperparamter configuration was executed, or a dictionary with keys as the hyperparameter configurations (as
        specified by A5 and A6) and values as the associated list of indices.
    C7. get_rlv_feat()
        :return: A list of indices of the relevant feature subset union selected by the two methods if only one
        hyperparamter configuration was executed, or a dictionary with keys as the hyperparameter configurations (as
        specified by A5 and A6) and values as the associated list of indices.

    References
    ----------
    1. BorutaPy Python package. https://github.com/scikit-learn-contrib/boruta_py.
    2. Kursa, M. B., & Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. Journal of Statistical
       Software, 36(11), 1–13. https://doi.org/10.18637/jss.v036.i11.
    """
    def __init__(self,
                 L_estimator: LogisticRegressionCV = LogisticRegressionCV(Cs=100, cv=5, penalty='l1', solver='saga',
                                                     max_iter=1000, n_jobs=-1, random_state=42),
                 L_beta: Union[float, int] = 0,
                 RF_estimator: RandomForestClassifier = RandomForestClassifier(max_depth=7, n_jobs=-1),
                 B_n_estimators: Union[Literal['auto'], int] = 'auto',
                 B_perc: Union[int, list[int]] = 100,
                 B_alpha: Union[float, list[float]] = 0.05,
                 B_two_step: bool = True,
                 B_max_iter: int = 100,
                 B_vote_rule: Literal[0, 1, 2] = 0,
                 B_random_state: Optional[Union[int, np.random.Generator]] = None,
                 B_verbose: Literal[0, 1, 2] = 1):

        # Type and value check
        assert isinstance(L_estimator, LogisticRegressionCV), \
            (f"L_estimator must be a LogisticRegressionCV object from sklearn.linear_model. Now its type is "
             f"{type(L_estimator)}.")
        self.L_estimator = L_estimator
        try:
            L_beta = float(L_beta)
        except:
            raise TypeError(f'L_beta must be an integer or a float. Now its type is {type(L_beta)}.')
        assert L_beta >= 0, \
            f"L_beta must be a non-negative float. Now its value is {L_beta}."
        self.L_beta = L_beta
        assert isinstance(RF_estimator, RandomForestClassifier),\
            (f"RF_estimator must be a RandomForestClassifier object from sklearn.ensemble. Not its type is "
             f"{type(RF_estimator)}.")
        self.RF_estimator=RF_estimator
        # Rely on boruta_optimized.py to check the validity of other arguments
        self.B_model = BorutaClass(estimator=RF_estimator, n_estimators=B_n_estimators,
                                   perc=B_perc, alpha=B_alpha, two_step=B_two_step,
                                   max_iter=B_max_iter, vote_rule=B_vote_rule,
                                   random_state=B_random_state, verbose=B_verbose)

    def fit_linear(self, X: np.ndarray, y: np.ndarray):
        try:
            X = np.array(X)
        except:
            raise TypeError(f'X must be (convertible to) a numpy array. Now its type is {type(X)}.')
        try:
            y = np.array(y)
        except:
            raise TypeError(f'y must be (convertible to) a numpy array. Now its type is {type(y)}.')
        assert len(X.shape) == 2, f'X must be two-dimensional. Now its dimension is {X.shape}.'
        assert len(y.shape) == 1, f'y must be one-dimensional. Now its dimension is {y.shape}.'
        print("Fitting LogisticRegressionCV model...", flush=True)
        self.L_estimator.fit(X, y)

    def fit_nonlinear(self, X: np.ndarray, y: np.ndarray):
        try:
            X = np.array(X)
        except:
            raise TypeError(f'X must be (convertible to) a numpy array. Now its type is {type(X)}.')
        try:
            y = np.array(y)
        except:
            raise TypeError(f'y must be (convertible to) a numpy array. Now its type is {type(y)}.')
        assert len(X.shape) == 2, f'X must be two-dimensional. Now its dimension is {X.shape}.'
        assert len(y.shape) == 1, f'y must be one-dimensional. Now its dimension is {y.shape}.'
        print("Fitting Boruta model...", flush=True)
        self.B_model.fit(X, y)

    def fit_both(self, X: np.ndarray, y: np.ndarray):
        self.fit_linear(X, y)           # Rely on fit_linear to check the validity of X and y
        self.fit_nonlinear(X, y)        # Rely on fit_nonlinear to check the validity of X and y

    def get_linear_coef(self):
        if _is_fitted(self.L_estimator):
            return self.L_estimator.coef_[0]
        else:
            raise ValueError('The LogisticRegressionCV model has not been fitted.')

    def get_linear_rlv_feat(self):
        L_coef = self.get_linear_coef()
        L_selected = list(np.where(np.abs(L_coef) > self.L_beta)[0])
        return L_selected

    def get_nonlinear_rlv_feat(self):
        B_selected = self.B_model.get_rlv_feat()
        if len(B_selected.keys()) == 1:
            B_selected = list(B_selected.values())[0]
        return B_selected

    def get_rlv_feat(self):
        L_selected = self.get_linear_rlv_feat()
        B_selected = self.get_nonlinear_rlv_feat()
        if type(B_selected) == list:
            return sorted(set(L_selected).union(B_selected))
        else:
            LB_selected = {}
            for k in B_selected.keys():
                B_selected_sub = B_selected[k]
                LB_selected[k] = sorted(set(L_selected).union(B_selected_sub))
            return LB_selected

########################################################################################################################
# Define feature selection class for regression
########################################################################################################################


class FSRegressor(BaseEstimator, TransformerMixin):
    """
    A. Runtime parameters
    ---------------------
    A1. L_estimator: A LassoCV object from sklearn.linear_model.
        Estimate the feature subset linearly relevant to the target variable. See the original documentation for its
        runtime parameter setting.
        https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LassoCV.html
        Default setting: L_estimator=LassoCV(n_alphas=100, max_iter=1000, cv=5, n_jobs=-1, random_state=42)
    A2. L_beta: A non-negative float.
        Strictness of the linear feature selection process where larger values indicate a stricter criterion.
        Default setting: L_beta=0
    A3. RF_estimator: A RandomForestRegressor object from sklearn.ensemble.
        The embedded model in Boruta that aims to return the feature subset non-linearly relevant to the target variable.
        See the original documentation for its runtime parameter setting.
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        Default setting: RF_estimator=RandomForestRegressor(max_depth=7, n_jobs=-1)
    A4. B_n_estimators: a positive integer or 'auto'.
        The number of estimators in the chosen ensemble method in A3. If 'auto', the number of estimators is
        determined automatically based on the size of the feature set.
        Default setting: B_n_estimators='auto'
    A5. B_perc: A (list of) positive integer(s) in the close interval [1, 100].
        The percentile of the shadow importance in Boruta to be compared to for each true feature in each iteration.
        Default setting: B_perc=100
    A6. B_alpha: A (list of) float(s) in the open interval (0, 1).
        The level of significance in Boruta used to compare with the p-values in each two-sided statistical hypothesis
        test.
        Default setting: B_alpha=0.05
    A7. B_two_step: A boolean.
        Using Bonferroni correction for p-values if True or Benjamini-Hochberg FDR then Bonferroni correction otherwise
        in Boruta.
        Default setting: B_two_step=True
    A8. B_max_iter: A positive integer.
        Number of maximum iterations to run in Boruta.
        Default setting: B_max_iter=100
    A9. B_vote_rule: An integer in [0, 1, 2].
        Different voting rules in Boruta to resolve disagreement of rejection across different hyperparameter
        configurations specified in A3 and A4 for each iteration. See "Remark: Modifications to BorutaPy" in
        boruta_optimized.py for details.
        Default setting: B_vote_rule=0
    A10. B_random_state: An integer or a numpy RNG object.
         Random seed used by the random number generator in Boruta.
         Default setting: B_random_state=None
    A11. B_verbose: An integer in [0, 1, 2].
         Verbosity in Boruta. No logging if 0, displaying iteration numbers and number of trees in the Random Forest
         model used if 1, and together with the indices of the selected feature subset for each hyperparameter
         configuration if 2.
         Default setting: B_verbose=1

    B. Attributes
    -------------
    B1. B_model: A BorutaClass object with runtime parameters specified in A3-A11
    (A1-A3 are initialized as instance attributes.)

    C. Methods
    ----------
C1. fit_linear(X, y)
        Fit the LogisticRegressionCV model with (feature set X, target y) where
        :param X: A two-dimensional numpy array. Samples of the feature set with dimension as (n_samples, n_features).
        :param y: A one-dimensional numpy array. Samples of the target with dimension as (n_samples).
    C2. fit_nonlinear(X, y)
        Fit the Boruta model with (feature set X, target y) where
        :param X: A two-dimensional numpy array. Samples of the feature set with dimension as (n_samples, n_features).
        :param y: A one-dimensional numpy array. Samples of the target with dimension as (n_samples).
    C3. fit_both(X, y)
        Fit both LogisticRegressionCV and Boruta models with (feature set X, target y) where
        :param X: A two-dimensional numpy array. Samples of the feature set with dimension as (n_samples, n_features).
        :param y: A one-dimensional numpy array. Samples of the target with dimension as (n_samples).
    C4. get_linear_coef()
        :return: A list of linear coefficients in the fitted LogisticRegressionCV model.
    C5. get_linear_rlv_feat()
        :return: A list of indices of the relevant feature subset selected by the LogisticRegressionCV model.
    C6. get_nonlinear_rlv_feat()
        :return: A list of indices of the relevant feature subset selected by the Boruta model if only one
        hyperparamter configuration was executed, or a dictionary with keys as the hyperparameter configurations (as
        specified by A5 and A6) and values as the associated list of indices.
    C7. get_rlv_feat()
        :return: A list of indices of the relevant feature subset union selected by the two methods if only one
        hyperparamter configuration was executed, or a dictionary with keys as the hyperparameter configurations (as
        specified by A5 and A6) and values as the associated list of indices.
    References
    ----------
    1. BorutaPy Python package. https://github.com/scikit-learn-contrib/boruta_py
    2. Kursa, M. B., & Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. Journal of Statistical
       Software, 36(11), 1–13. https://doi.org/10.18637/jss.v036.i11
    """
    def __init__(self,
                 L_estimator: LassoCV = LassoCV(n_alphas=100, max_iter=1000, cv=5, n_jobs=-1, random_state=42),
                 L_beta: float =0,
                 RF_estimator=RandomForestRegressor(max_depth=7, n_jobs=-1),
                 B_n_estimators='auto',
                 B_perc=100,
                 B_alpha=0.05,
                 B_two_step=True,
                 B_max_iter=100,
                 B_vote_rule=0,
                 B_random_state=42,
                 B_verbose=1):

        # Type and value check
        assert isinstance(L_estimator, LassoCV), \
            (f"L_estimator must be a LassoCV object from sklearn.linear_model. Now its type is "
             f"{type(L_estimator)}.")
        self.L_estimator = L_estimator
        try:
            L_beta = float(L_beta)
        except:
            raise TypeError(f'L_beta must be an integer or a float. Now its type is {type(L_beta)}.')
        assert L_beta >= 0, \
            f"L_beta must be a non-negative float. Now its value is {L_beta}."
        self.L_beta = L_beta
        assert isinstance(RF_estimator, RandomForestRegressor), \
            (f"RF_estimator must be a RandomForestRegressor object from sklearn.ensemble. Now its type is "
             f"{type(RF_estimator)}.")
        self.RF_estimator = RF_estimator
        # Rely on boruta_optimized.py to check the validity of other arguments
        self.B_model = BorutaClass(estimator=RF_estimator, n_estimators=B_n_estimators,
                                   perc=B_perc, alpha=B_alpha, two_step=B_two_step,
                                   max_iter=B_max_iter, vote_rule=B_vote_rule,
                                   random_state=B_random_state, verbose=B_verbose)

    def fit_linear(self, X: np.ndarray, y: np.ndarray):
        try:
            X = np.array(X)
        except:
            raise TypeError(f'X must be (convertible to) a numpy array. Now its type is {type(X)}.')
        try:
            y = np.array(y)
        except:
            raise TypeError(f'y must be (convertible to) a numpy array. Now its type is {type(y)}.')
        assert len(X.shape) == 2, f'X must be two-dimensional. Now its dimension is {X.shape}.'
        assert len(y.shape) == 1, f'y must be one-dimensional. Now its dimension is {y.shape}.'
        print("Fitting LassoCV model...", flush=True)
        self.L_estimator.fit(X, y)

    def fit_nonlinear(self, X: np.ndarray, y: np.ndarray):
        try:
            X = np.array(X)
        except:
            raise TypeError(f'X must be (convertible to) a numpy array. Now its type is {type(X)}.')
        try:
            y = np.array(y)
        except:
            raise TypeError(f'y must be (convertible to) a numpy array. Now its type is {type(y)}.')
        assert len(X.shape) == 2, f'X must be two-dimensional. Now its dimension is {X.shape}.'
        assert len(y.shape) == 1, f'y must be one-dimensional. Now its dimension is {y.shape}.'
        print("Fitting Boruta model...", flush=True)
        self.B_model.fit(X, y)

    def fit_both(self, X: np.ndarray, y: np.ndarray):
        self.fit_linear(X, y)           # Rely on fit_linear to check the validity of X and y
        self.fit_nonlinear(X, y)        # Rely on fit_nonlinear to check the validity of X and y

    def get_linear_coef(self):
        if _is_fitted(self.L_estimator):
            return self.L_estimator.coef_
        else:
            raise ValueError('The LassoCV model has not been fitted.')

    def get_linear_rlv_feat(self):
        L_coef = self.get_linear_coef()
        L_selected = list(np.where(np.abs(L_coef) > self.L_beta)[0])
        return L_selected

    def get_nonlinear_rlv_feat(self):
        B_selected = self.B_model.get_rlv_feat()
        return list(B_selected.values())[0] if len(B_selected.keys()) == 1 else B_selected

    def get_rlv_feat(self):
        L_selected = self.get_linear_rlv_feat()
        B_selected = self.get_nonlinear_rlv_feat()
        if type(B_selected) == list:
            return sorted(set(L_selected).union(B_selected))
        else:
            LB_selected = {}
            for k in B_selected.keys():
                B_selected_sub = B_selected[k]
                LB_selected[k] = sorted(set(L_selected).union(B_selected_sub))
            return LB_selected

########################################################################################################################
