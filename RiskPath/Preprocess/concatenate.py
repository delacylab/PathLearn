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
from itertools import chain
from typing import Literal, Union

########################################################################################################################
# Define the timestamp concatenating operation as a class
########################################################################################################################


class TSConcatenate:
    """
    This class is designed to preprocess non-square datasets (in terms of irregular timestamps). Notice that a feature
    can have no samples in a given timestamp. In that case, we impute the no-sample feature with surrogate data, either
    as a constant or from samples in other timestamps.

    A. Runtime parameters
    ---------------------
    A1. verbose: an integer in [0, 1]
        Verbosity. No logging if 0, and logging about number of features needed imputation if 1.
        Default setting: verbose=1

    B. Attributes
    -------------
    B1. data_input_dict: A dictionary.
        A dictionary with keys as timestamps and values as the list of pandas.DataFrame input using the
        load_data_in_order method in C1.
    B2. feat_dict: A dictionary.
        A dictionary with keys as timestamps and values as the list of feature names associated to the pandas.DataFrame
        in B1.
    B3. feat_subset_dict: A dictionary.
        A dictionary with keys as timestamps and values as the list of feature subset names associated to the
        pandas.DataFrame in B1. It is useful only when the method C2 load_fs_result_in_order has been called.
    B4. feat_union: A list.
        A list of unique feature names from all timestamps, or its subset restricted to B3 if the method C2
        load_fs_result_in_order has been called.
    B5. data_output_dict: A dictionary.
        A dictionary with keys as timestamps and values as the list of pandas.DataFrame imputed after calling a
        concatenate method in C3.
    B6. nan_feature_dict: A dictionary.
        A dictionary with keys as timestamps and values as the indices of features needed imputation.
    (A1 is initialized as instance attributes.)

    C. Methods
    ----------
    C1. load_data_in_order(list_of_df)
        Load the pandas.DataFrame of each timestamp in order, from earliest to latest.
        Remark: All indexing/identifier columns must be removed before using this method.
        Remark: All pandas.DataFrame must have the same sample size (but not necessarily the same set of features).
        :param list_of_df: a list of pandas.DataFrame, each with rows as samples and columns as features.
    C2. load_fs_result_in_order(list_of_fs_result)
        Load the list of feature subset indices (e.g., output of feature selection) of each timestamp in order, from
        earliest to latest.
        :param list_of_fs_result: a list of lists where each sub-list is a list of feature indices.
    C3. concatenate(strategy)
        :param strategy: an integer, a float, or a string in ['mean', 'median', 'backward', 'forward'].
        The method of data imputation when a feature has no sample in a given timestamp. If int or float, all samples
        are filled with the specified value. If 'mean' (or 'median'), each sample is computed from the mean (or median)
        across timestamps where the feature has samples. If 'backward' (or 'forward), each sample is copied from the
        samples of the closest earlier (or later) timestamp. But if no such timestamp exists, each sample is copied from
        the samples of the closest later (or earlier) timestamp.
        Default setting: method='mean'
        :return: self.data_output_dict. A dictionary with keys as timestamp indices and values as pandas.DataFrame with
                 non-existent features filled with surrogate data.
    C4. get_numpy(dim, suffix)
        :param dim: an integer in [2, 3], representing the dimension of the output numpy array.
               Default setting: dim=3
        :param suffix_sep: string separator used to identify the timestamp of a feature such that its resulting name
               is {feature name}{suffix_sep}{timestamp}.
               Default setting: suffix_sep='_t'
        :return:
        (a) A numpy array encoding self.data_output_dict with dimension of
            (sample size, number of timestamps, number of features) if dim=2, or
            (sample size, number of timestamps * number of features) if dim=3.
        (b) A list of feature names associated with the output numpy array in (a).
        (c) A list of feature names that have been imputed.
        """

    def __init__(self,
                 verbose: Literal[0, 1] = 1):
        # Type and value check
        assert verbose in [0, 1], \
            f"verbose must be in [0, 1]. Now its value is {verbose}."
        self.verbose = verbose
        self.data_input_dict = None
        self.feat_dict = None
        self.feat_subset_dict = None
        self.feat_union = None
        self.data_output_dict = {}
        self.nan_feature_dict = {}

    def load_data_in_order(self, list_of_df: list[pd.DataFrame]):
        # Type and value check
        assert isinstance(list_of_df, list), \
            f'list_of df must be a list. Now its type is {type(list_of_df)}.'
        sample_size, data_dict, feat_dict = None, {}, {}
        for df_idx, df in enumerate(list_of_df):
            assert isinstance(df, pd.DataFrame), \
                f'Each element in list_of_df must be a pandas.DataFrame.'
            if df_idx == 0:
                sample_size = df.shape[0]
            else:
                assert df.shape[0] == sample_size, \
                    f'Each element must be a pandas.DataFrame with the same sample size.'
            data_dict[df_idx] = df
            feat_dict[df_idx] = list(df.columns)

        # Update attributes
        self.data_input_dict, self.feat_dict = data_dict, feat_dict
        self.feat_subset_dict = {ts: None for ts in data_dict.keys()}
        self.feat_union = sorted(set(chain.from_iterable(feat_dict.values())))

        # Identify no-sample features in each timestamp
        for ts_i, data_i in self.data_input_dict.items():
            self.nan_feature_dict[ts_i] = sorted(set(self.feat_union).difference(data_i.columns))

    def load_fs_result_in_order(self, list_of_fs_result: list[list[int]]):
        # Runtime, type, and value check
        assert self.data_input_dict is not None, \
            f"Method load_data_in_order must be used first."
        assert isinstance(list_of_fs_result, list), \
            f'list_of_df must be a list. Now its type is {type(list_of_fs_result)}.'
        assert len(list_of_fs_result) == len(self.data_input_dict), \
            (f'The length of list_of_fs_result (= {len(list_of_fs_result)}) must be the same as the length of '
             f'list_of_df (={len(self.data_input_dict)}) used in load_data_in_order.')
        feat_subset_dict = {}
        for idx, (ts_i, feat_list) in enumerate(self.feat_dict.items()):
            fs_result = list_of_fs_result[idx]
            assert isinstance(fs_result, list), \
                f'list_of_df must be a list of lists.'
            try:
                fs_result = np.array(fs_result, dtype=int)
            except:
                raise TypeError(f"Each feature index must be an integer.")
            try:
                fs_subset = [feat_list[i] for i in fs_result]
            except:
                raise IndexError(f'The {ts_i}-th list of indices contains a feature index out of the range.')
            feat_subset_dict[ts_i] = fs_subset

        # Update attributes
        self.feat_subset_dict = feat_subset_dict
        self.feat_union = sorted(set(chain.from_iterable(feat_subset_dict.values())))
        for ts_i, data_i in self.data_input_dict.items():
            self.nan_feature_dict[ts_i] = sorted(set(self.feat_union).difference(data_i.columns))

    def concatenate(self,
                    strategy: Union[int, float, Literal['mean', 'median', 'forward', 'backward']] = 'mean'):
        # Runtime, type, and value check
        assert (isinstance(strategy, int) or isinstance(strategy, float) or
                strategy in ['mean', 'median', 'forward', 'backward']), \
            (f"method must be an integer, a float, or a string in ['mean', 'median', 'forward', 'backward']. "
             f"Now its value is {strategy}.")
        assert self.data_input_dict is not None, \
            f"Method load_data_in_order must be called first."

        for ts_i, data_i in self.data_input_dict.items():
            self.data_output_dict[ts_i] = data_i.copy()                   # Update every time .concatenate is called
            nan_feats = sorted(set(self.feat_union).difference(data_i.columns))
            if self.verbose == 1:
                print(f"For the {ts_i}-th timestamp, {len(nan_feats)} features will be imputed by surrogate data "
                      f"using method={strategy}.", flush=True)

            if not isinstance(strategy, str):                             # Constant padding
                for col in nan_feats:
                    self.data_output_dict[ts_i].loc[:, col] = strategy

            elif strategy in ['mean', 'median']:                          # Mean/median padding
                for col in nan_feats:
                    resource = pd.DataFrame(None)
                    for ts_other in self.feat_dict.keys():
                        if col in self.data_input_dict[ts_other].columns:
                            resource[f'{col}_{ts_other}'] = self.data_input_dict[ts_other][col]
                    resource_np = resource.values
                    method_func = np.nanmean if strategy == 'mean' else np.nanmedian
                    self.data_output_dict[ts_i].loc[:, col] = method_func(resource_np, axis=1)

            else:                                                       # Forward/backward padding
                for col in nan_feats:
                    ts_others = [ts_j for ts_j in self.feat_dict.keys()
                                 if col in self.data_input_dict[ts_j].columns]
                    ts_others_backward = [ts_j for ts_j in ts_others if ts_j < ts_i]
                    ts_others_forward = [ts_j for ts_j in ts_others if ts_j > ts_i]
                    ts_j_backward = max(ts_others_backward) if len(ts_others_backward) != 0 else np.nan
                    ts_j_forward = min(ts_others_forward) if len(ts_others_forward) != 0 else np.nan
                    ts_j = np.nanmin([ts_j_backward, ts_j_forward]) if strategy == 'backward' \
                        else np.nanmax([ts_j_backward, ts_j_forward])
                    self.data_output_dict[ts_i].loc[:, col] = self.data_input_dict[ts_j][col]

            self.data_output_dict[ts_i] = self.data_output_dict[ts_i].reindex(self.feat_union, axis=1)
        return self.data_output_dict

    def get_result(self, dim: Literal[2, 3] = 3, sep: str = '_t'):
        # Runtime, type, and value check
        assert self.data_output_dict != {}, \
            f"Method concatenate must be called first."
        assert dim in [2, 3], \
            f"dim must be in [2, 3]. Now it is {dim}."
        assert isinstance(sep, str), \
            f"sep must be a string. Now its type is {type(sep)}."

        # Obtain the concatenated output as a 2- or 3-dimensional numpy array
        X_output = np.stack([d.to_numpy() for d in self.data_output_dict.values()], axis=1) if dim == 3 \
            else np.concatenate(list(self.data_output_dict.values()), axis=1)

        # Obtain the full feature list
        feat_list = self.feat_union if dim == 3 \
            else [f'{feat}{sep}{ts_i}' for ts_i in self.data_output_dict.keys() for feat in self.feat_union]

        # Identify the imputed features
        imputed = [feat + sep + f'{ts_i}' for ts_i, feat_list in self.nan_feature_dict.items() for feat in feat_list]

        return X_output, feat_list, imputed

########################################################################################################################
