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
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from tslearn.generators import random_walks, random_walk_blobs
from typing import Optional, Union

########################################################################################################################
# Define a time-series data simulation function for classification
########################################################################################################################


def make_ts_classification(n_samples_per_class: int,
                           n_timestamps: int,
                           n_features: int,
                           n_informative: int,
                           n_classes: int,
                           noise_level: int = 1,
                           shuffle: bool = False,
                           random_state: Optional[Union[int, np.random.Generator]] = None):
    """
    This function borrows the time-series generator function from tslearn.generators.random_walk_blobs and extends it
    to allow irrelevant features. For the original function from tslearn, see
    https://tslearn.readthedocs.io/en/stable/gen_modules/generators/tslearn.generators.random_walk_blobs.html.
    The format mimics the commonly used make_classification function from sklearn.datasets. See
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

    :param n_samples_per_class: A positive integer.
           The sample size per class of the simulated data.
    :param n_timestamps: A positive integer.
           The number of timestamps of the simulated data.
    :param n_features: A positive integer.
           The number of features (per timestamp) of the simulated data.
    :param n_informative: A positive integer not greater than n_features.
           The number of informative features (per timestamp) of the simulated data.
    :param n_classes: A positive integer greater than 1.
           The number of classes in the target.
    :param noise_level: A non-negative integer or float.
           Standard deviation of the white noise.
           Default setting: noise_level=1
    :param shuffle: A boolean.
           Shuffle the feature set if True.
           Default setting: shuffle=False
    :param random_state: An integer or None.
           Random seed used by the random number generator.
    :return:
    (a) X: A three-dimensional numpy array with dimension of (n_classes * n_samples, n_timestamps, n_features).
    (b) y: A one-dimensional numpy array with dimension of (n_classes * n_samples, ).
    """

    # Type and value check
    assert isinstance(n_samples_per_class, int), \
        f"n_samples_per_class must be an integer. Now its type is {type(n_samples_per_class)}."
    assert n_samples_per_class >= 1, \
        f"n_samples_per_class must be a positive integer. Now its value is {n_samples_per_class}."
    assert isinstance(n_timestamps, int), \
        f"n_timestamps must be an integer. Now its type is {type(n_timestamps)}."
    assert n_timestamps >= 1, \
        f"n_timestamps must be at least 1. Now its value is {n_timestamps}."
    assert isinstance(n_features, int), \
        f"n_features must be an integer. Now its type is {type(n_features)}."
    assert n_features >= 1, \
        f"n_features must be at least 1. Now its value is {n_features}."
    assert isinstance(n_informative, int), \
        f"n_informative must be an integer. Now its type is {type(n_informative)}."
    assert n_features >= n_informative >= 1, \
        f"n_informative must be in the range [1, n_features]. Now its value is {n_informative}."
    assert isinstance(n_classes, int), \
        f"n_classes must be an integer. Now its type is {type(n_classes)}."
    assert n_classes >= 2, f"n_classes must be at least 2. Now its value is {n_classes}."
    try:
        noise_level = float(noise_level)
    except:
        raise TypeError(f"noise level must be (convertible to) a float. Now its type is {type(noise_level)}.")
    assert noise_level >= 0, \
        f"noise_level must be non-negative. Now its value is {noise_level}."
    assert isinstance(shuffle, bool), \
        f'shuffle must be a boolean. Now its type is {type(shuffle)}.'
    rng = check_random_state(random_state)               # Rely on sklearn to check if random_state is valid

    # Use tslearn.generators.random_walk_blobs to simulate base data
    X, y = random_walk_blobs(n_ts_per_blob=n_samples_per_class,
                             sz=n_timestamps,
                             d=n_informative,
                             n_blobs=n_classes,
                             noise_level=noise_level,
                             random_state=rng)

    # Create samples of uninformative features from random noises and concatenate with the base data
    X_uninformative = rng.randn(X.shape[0], X.shape[1], n_features-n_informative)
    X = np.concatenate([X, X_uninformative], axis=2)
    if shuffle:
        indices = rng.permutation(X.shape[2])
        X = np.take_along_axis(X, indices[None, None, :], axis=2)
    return X, y

########################################################################################################################
# Define a time-series data simulation function for regression
########################################################################################################################


def make_ts_regression(n_samples: int,
                       n_timestamps: int,
                       n_features: int,
                       n_informative: int,
                       shuffle: bool = False,
                       random_state: Optional[Union[int, np.random.Generator]] = None):
    """
    This function borrows the time-series generator function from tslearn.generators.random_walk and extends it to
    allow irrelevant features. For the original function from tslearn, see
    https://tslearn.readthedocs.io/en/stable/gen_modules/generators/tslearn.generators.random_walks.html
    The format mimics the commonly used make_classification function from sklearn.datasets. See
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

    :param n_samples: A positive integer.
           The sample size of the simulated data.
    :param n_timestamps: A positive integer.
           The number of timestamps of the simulated data.
    :param n_features: A positive integer.
           The number of features (per timestamp) of the simulated data.
    :param n_informative: A positive integer not greater than n_features.
           The number of informative features (per timestamp) of the simulated data.
    :param shuffle: A boolean.
           Shuffle the feature set if True.
           Default setting: shuffle=False
    :param random_state: An integer or None.
           Random seed used by the random number generator.
    :return:
    (a) X: A three-dimensional numpy array with dimension of (n_samples, n_timestamps, n_features).
    (b) y: A one-dimensional numpy array with dimension of (n_samples, ).
    """

    # Type and value check
    assert isinstance(n_samples, int), \
        f"n_samples_per_class must be an integer. Now its type is {type(n_samples)}."
    assert n_samples >= 1, \
        f"n_samples must be a positive integer. Now its value is {n_samples}."
    assert isinstance(n_timestamps, int), \
        f"n_timestamps must be an integer. Now its type is {type(n_timestamps)}."
    assert n_timestamps >= 1, \
        f"n_timestamps must be at least 1. Now its value is {n_timestamps}."
    assert isinstance(n_features, int), \
        f"n_features must be an integer. Now its type is {type(n_features)}."
    assert n_features >= 1, \
        f"n_features must be at least 1. Now its value is {n_features}."
    assert isinstance(n_informative, int), \
        f"n_informative must be an integer. Now its type is {type(n_informative)}."
    assert n_features >= n_informative >= 1, \
        f"n_informative must be in the range [1, n_features]. Now its value is {n_informative}."
    assert isinstance(shuffle, bool), \
        f'shuffle must be a boolean. Now its type is {type(shuffle)}.'
    rng = check_random_state(random_state)      # Rely on sklearn to check if random_state is valid

    # Use tslearn.generators.random_walk to simulate base data
    X = random_walks(n_ts=n_samples, sz=n_timestamps, d=n_informative, random_state=rng)

    # Generate the target
    X_reduced = X.mean(axis=2)
    weights = np.linspace(0.1, 1, X_reduced.shape[1])
    y = np.sum(X_reduced * weights, axis=1)

    # Create samples of uninformative features from random noises and concatenate with the base data
    X_uninformative = rng.randn(X.shape[0], X.shape[1], n_features-n_informative)
    X = np.concatenate([X, X_uninformative], axis=2)
    if shuffle:
        indices = rng.permutation(X.shape[2])
        X = np.take_along_axis(X, indices[None, None, :], axis=2)
    return X, y

########################################################################################################################
# Define a sample synthetic time-series dataset for RiskPath tutorial
########################################################################################################################


def sample_dataset_1():
    """
    Generate a synthetic time-series dataset for RiskPath tutorial.
    - The dimension of the full feature set X is (1000 samples, 10 timestamps, 30 features).
    - The dimension of the full binary target y is (1000 samples,).
    - 30 features in X are important relative to y.
    - Train-test ratio of 7:3 so that there are 700 (resp. 300) samples in the training (resp. test) partition.
    This function borrows the time-series generator function from tslearn.generators.random_walk_blobs and extends it
    to allow irrelevant features. For the original function from tslearn, see
    https://tslearn.readthedocs.io/en/stable/gen_modules/generators/tslearn.generators.random_walk_blobs.html.

    :return:
    (a) X_train: a 3-dimensional numpy array.
        The training partition of X with dimension (700 samples, 10 timestamps, 30 features)
    (b) X_test: a 3-dimensional numpy array.
        The test partition of X with dimension (300 samples, 10 timestamps, 30 features)
    (c) y_train: a 1-dimensional numpy array.
        The training partition of y with dimension (700 samples,).
    (d) y_test: a 1-dimensional numpy array.
        The test partition of y with dimension (300 samples,).
    """
    X, y = make_ts_classification(n_samples_per_class=500,
                                  n_timestamps=10,
                                  n_features=30,
                                  n_informative=10,
                                  n_classes=2,
                                  noise_level=10,
                                  random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    feat_names = [f'X{j}' for j in range(X.shape[2])]
    return X_train, X_test, y_train, y_test, feat_names

########################################################################################################################
# Define a sample synthetic time-series dataset for RiskPath full tutorial
########################################################################################################################


def sample_dataset_2():
    """
    Generate a synthetic time-series dataset for RiskPath tutorial (with feature selection and timestamp concatenation).
    - The dimension of the full feature set X is (2000 samples, 5 timestamps, 30 features).
    - The dimension of the full binary target y is (2000 samples,).
    - 100 features in X are important relative to y.
    - For each timestamp, 5~15% randomly sampled features in X have no samples.
    - Train-test ratio of 7:3 so that there are 1400 (resp. 600) samples in the training (resp. test) partition.
    This function borrows the time-series generator function from tslearn.generators.random_walk_blobs and extends it
    to allow irrelevant features. For the original function from tslearn, see
    https://tslearn.readthedocs.io/en/stable/gen_modules/generators/tslearn.generators.random_walk_blobs.html.

    :return:
    (a) X_train_list: a list of five 2-dimensional Pandas.DataFrame.
        Each i-th element is the training partition of X with dimension (1400 samples, j features) in the i-th
        timestamp where j refers to the number of features with no missing data.
    (b) X_test_list: a list of five 2-dimensional Pandas.DataFrame.
        Each i-th element is the test partition of X with dimension (600 samples, j features) in the i-th timestamp
        where j refers to the number of features with no missing data.
    (c) y_train: a 1-dimensional numpy array.
        The training partition of y with dimension (1400 samples,).
    (d) y_test: a 1-dimensional numpy array.
        The test partition of y with dimension (600 samples,).
    """
    X, y = make_ts_classification(n_samples_per_class=1000,
                                  n_timestamps=5,
                                  n_features=30,
                                  n_informative=10,
                                  n_classes=2,
                                  noise_level=5,
                                  random_state=42)
    rng = np.random.default_rng(42)
    while True:         # Ensures that no feature has no samples throughout all the timestamps
        for t_i in range(X.shape[1]):
            # For each timestamp, randomly sample 5~15% of the features to have no samples.
            nan_percent = rng.integers(low=5, high=16) / 100
            nan_feat = rng.choice(X.shape[2], size=int(nan_percent*X.shape[2]), replace=False)
            X[:, t_i, nan_feat] = np.nan
        if not np.any(np.all(np.isnan(X), axis=(0, 1))):
            break

    # Partition the feature set and target into training and test partitions with a 7:3 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Convert to a pandas.DataFrame for each timestamp
    feat_names = [f'X{j}' for j in range(X.shape[2])]
    X_train_list, X_test_list = [], []
    for t_i in range(X.shape[1]):
        X_train_i = pd.DataFrame(X_train[:, t_i, :], columns=feat_names)
        X_test_i = pd.DataFrame(X_test[:, t_i, :], columns=feat_names)
        non_nan_feat = np.where(X_train_i.isna().sum(axis=0).values != X_train_i.shape[0])[0]
        X_train_list.append(X_train_i.iloc[:, non_nan_feat])
        X_test_list.append(X_test_i.iloc[:, non_nan_feat])
    return X_train_list, X_test_list, y_train, y_test, feat_names

########################################################################################################################
