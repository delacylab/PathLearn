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
# Import packages
########################################################################################################################
import numpy as np
import pandas as pd
from .metrics import classify_metrics, classify_AUROC
from sklearn.model_selection import StratifiedKFold
from time import time
from typing import Literal, Optional, Union

########################################################################################################################
# Define a simple embedded procedure to train (with cross-validation) and evaluate classical machine learning algorithms
########################################################################################################################


class Benchmark_Classifier:
    """
    A benchmark classifier for sklearn models.

    A. Runtime parameters
    ---------------------
    A1. base_model: A sklearn estimator object.
    Presumably, base_model is one of the following.
    - sklearn.linear_model.LogisticRegression()
    - sklearn.ensemble.RandomForestClassifier()
    - sklearn.svm.SVC(probability=True)
    Any other estimator object with the method of .predict_proba can also be used.

    B. Attributes
    -------------
    B1. best_model: The best fitted model.
    B2. model_performance: A dictionary with keys as names of performance metrics and values as the values of the
                           metrics in the best fitted model in B1.
    B3. model_performance_df: A one-row pandas.DataFrame converted from the dictionary in B2.
    B4. TFPR_dict: A dictionary with keys as partitions' strings (e.g., 'Train', 'Val') and values as
                   [list of TPR values, list of FPR values].
    (A1 is initialized as an instance attribute.)

    C. Methods
    ----------
    C1. fit(X, y, cv, val_metric, random_state)
    Fit the base model with cross-validation.
    :param X: A 2-dimensional numpy array.
           Samples of the feature set with dimension as (sample size, number of features).
    :param y: A 1-dimensional numpy array.
           Samples of the target with dimension as (sample size,).
    :param cv: An integer.
           The number of folds used in the cross-validation splitting strategy.
           Default setting: cv=5
    :param val_metric: A string in B1.
           Specify the metric used to judge which model in the cross-validation process is the best.
           Default setting: val_metric='AUROC'
    :param random_state: An integer, a numpy.random.RandomState instance, or None.
           The seed of the random number generator used in the cross-validation process.
           Default setting: random_state=None
    C2. evaluate(X, y, prefix)
        Evaluate the models fitted in C2 with the feature set X and the target y.
        :param X: A 2- or 3-dimensional numpy array.
               Samples of the feature set with dimension as (sample size, number of features).
        :param y: A 1-dimensional numpy array.
               Samples of the target with dimension as (sample size,).
        :param prefix: A string.
               The prefix used in the string for each performance metric.
               Default setting: prefix='Test'
    C3. get_performance()
        Obtain a pandas.DataFrame summarizing the performance statistics of the best-fit model.
        :return: A one-row Pandas.DataFrame of the performance metrics and values.
    """

    def __init__(self,
                 base_model):
        self.best_model = None
        self.model_performance = None
        self.model_performance_df = None
        self.TFPR_dict = {}
        self.base_model = base_model

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            cv: int = 5,
            val_metric: Literal['AUROC', 'Accuracy', 'F1', 'NLL', 'Precision', 'Recall', 'Specificity',
                                'loss'] = 'AUROC',
            random_state: Optional[Union[int, np.random.RandomState]] = None):

        # Type and value check
        try:
            X = np.array(X)
        except TypeError:
            raise TypeError(f'X must be (convertible to) a numpy.ndarray. Now its type is {type(X)}.')
        try:
            y = np.array(y)
        except TypeError:
            raise TypeError(f'y must be (convertible to) a numpy.ndarray. Now its type is {type(y)}.')
        assert len(X.shape) == 2, f'X must be 2-dimensional. Now its dimension is {X.shape}.'
        assert len(y.shape) == 1, f'y must be 1-dimensional. Now its dimension is {y.shape}.'
        assert isinstance(cv, int), \
            f'cv has to be a positive integer. Now its type is {type(cv)}.'
        assert cv >= 1, \
            f'cv has to be a positive integer. Now it is {cv}.'
        metrics_list = ['AUROC', 'Accuracy', 'F1', 'NLL', 'Precision', 'Recall', 'Specificity', 'loss']
        assert val_metric in metrics_list, \
            f'val_metric must be in {metrics_list}. Now it is {val_metric}.'

        skf = StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=True)
        best_val_score, best_model, best_performance = float('-inf'), None, None
        best_train_TFPR, best_val_TFPR = None, None
        train_tfpr_pair, val_tfpr_pair = [], []  # Temporary storage of TPR and FPR values

        start_time = time()
        for k_idx, (train, val) in enumerate(skf.split(X, y), 1):
            X_train, X_val = np.take(X, train, axis=0), np.take(X, val, axis=0)
            y_train, y_val = np.take(y, train), np.take(y, val)

            M = self.base_model
            M.fit(X_train, y_train)
            fit_result = {}
            for (X_, y_, prefix) in [(X_train, y_train, 'Train_'), (X_val, y_val, 'Val_')]:
                y_pred_ = M.predict(X_)
                y_pred_prob_ = M.predict_proba(X_)[:, 1]
                fit_result |= classify_metrics(y_, y_pred_, prefix=prefix)
                auroc_, tpr_, fpr_ = classify_AUROC(y_, y_pred_prob_, prefix=prefix, auroc_only=False)
                fit_result |= auroc_
                if prefix == 'Train_':
                    train_tfpr_pair = [tpr_, fpr_]
                else:
                    val_tfpr_pair = [tpr_, fpr_]
            score_name = f'Val_{val_metric}'
            score = -fit_result[score_name] if val_metric in ['NLL', 'loss'] else fit_result[score_name]

            if score > best_val_score:
                best_val_score, best_model, best_performance = score, M, dict(
                    sorted(fit_result.items()))
                best_train_TFPR, best_val_TFPR = train_tfpr_pair, val_tfpr_pair
        end_time = time()
        elapsed = end_time - start_time

        self.best_model = best_model
        self.model_performance = best_performance | {'Elapsed_train_time': elapsed}
        self.TFPR_dict['Train'] = best_train_TFPR
        self.TFPR_dict['Val'] = best_val_TFPR

    def evaluate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 prefix: str = 'Test'):

        # Type and value check
        try:
            X = np.array(X)
        except TypeError:
            raise TypeError(f'X must be (convertible to) a numpy.ndarray. Now its type is {type(X)}.')
        try:
            y = np.array(y)
        except TypeError:
            raise TypeError(f'y must be (convertible to) a numpy.ndarray. Now its type is {type(y)}.')
        assert len(X.shape) == 2, f'X must be 2-dimensional. Now its dimension is {X.shape}.'
        assert len(y.shape) == 1, f'y must be 1-dimensional. Now its dimension is {y.shape}.'
        assert isinstance(prefix, str), \
            f"prefix must be a string. Now its type is {type(prefix)}."

        M = self.best_model
        start_time = time()
        y_pred_ = M.predict(X)
        end_time = time()
        y_pred_prob_ = M.predict_proba(X)[:, 1]
        fit_result = self.model_performance
        fit_result |= classify_metrics(y, y_pred_, prefix=f'{prefix}_')
        auroc_, tpr_, fpr_ = classify_AUROC(y, y_pred_prob_, prefix=f'{prefix}_', auroc_only=False)
        fit_result |= auroc_
        fit_result |= {'Elapsed_test_time': end_time - start_time}
        self.model_performance = fit_result
        self.TFPR_dict[prefix] = [tpr_, fpr_]

    def get_performance(self):
        df = pd.DataFrame([], columns=self.model_performance.keys())
        for k in sorted(self.model_performance.keys()):
            df.loc[0, k] = self.model_performance[k]
        self.model_performance_df = df
        return df

########################################################################################################################


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=100, n_informative=10,
                               n_classes=2, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)
    M = Benchmark_Classifier(base_model=LogisticRegression())
    M.fit(X_train, y_train)
    M.evaluate(X_test, y_test)
    print(M.get_performance())
