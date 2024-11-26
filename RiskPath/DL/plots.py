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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import warnings
from matplotlib.cm import get_cmap
from PIL import Image
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Union
plt.style.use('seaborn-v0_8-darkgrid')          # Consider 'ggplot' or 'Solarize_Light2'

########################################################################################################################
# Define the plotting function for models' performance statistics
########################################################################################################################

def plot_performance(df: pd.DataFrame,
                     var_column: str,
                     prefixes: list[str],
                     metric: str,
                     sep: str = '_',
                     title: Optional[str] = None,
                     filename: Optional[str] = None,
                     rename_dict: Optional[Dict[str, str]] = None):
    """
    Plot the performance statistics across different parameter settings and partitions.
    :param df: A Pandas.DataFrame.
           Performance statistics across different parameter settings and partitions.
    :param var_column: A string.
           The column name in df that corresponds to the tuned parameter.
    :param prefixes: A list of strings.
           Strings encoding the partitions or extra experiments (e.g., 'Train', 'Val', 'Test')
    :param metric: A string.
           The string of the metric of interest to evaluate and compare models.
    :param sep: A string.
           The string separating the prefix and metric in the column of df that encodes the metric values.
           Default setting: sep='_'
    :param title: A string or None.
           The title of the plot if a string is provided, and no title provided if None.
           Default setting: title=None
    :param filename: A string or None. If not None, a PNG file with the specified filename will be created in the
           current working directory.
           Default setting: filename=None
    :param rename_dict: A dictionary (with keys and values as strings) or None.
           If not None, it functions as a dictionary to map strings from var_column, prefixes, and metric to other
           strings that will be shown in the plot.
           Example: rename_dict={'n_units': 'Number of hidden units', 'Val': 'Validation', 'loss': 'Cross Entropy'}
           Default setting: rename_dict=None
    :return:
    (a) A plot displayed to the IDE.
    If filename is not None:
    (b) A file named {filename}.png saved to the current working directory.
    """

    # Type and value check
    assert isinstance(df, pd.DataFrame), \
        f"df must be a Pandas.DataFrame. Now its type is {type(df)}."
    assert isinstance(var_column, str), \
        f"var_column must be a string. Now its type is {type(var_column)}."
    assert var_column in df.columns, \
        f"val_column must be a column in df."
    assert df[var_column].nunique() == df.shape[0], \
        f"Elements in val_column must not have duplicates."
    try:
        df[var_column] = df[var_column].astype('float')
        df.sort_values(by=var_column, axis=0, key=pd.to_numeric, ignore_index=True, inplace=True)
    except:
        raise TypeError('val_column must be a numeric column.')
    assert isinstance(prefixes, list), \
        f'prefixes must be a list. Now its type is {type(prefixes)}.'
    assert len(prefixes) <= 6, \
        f"Due to readability, the length of prefixes must not exceed 6. Now its length is {len(prefixes)}."
    assert all([isinstance(p, str) for p in prefixes]), \
        f"All elements in prefixes must be a string."
    assert isinstance(metric, str), \
        f"metric must be a string. Now its type is {type(metric)}."
    assert isinstance(sep, str), \
        f"sep must be a (possibly empty) string. Now its type is {type(sep)}."
    if title is not None:
        assert isinstance(title, str), \
            f"title must be a string or None. Now its type is {type(title)}."
    if filename is not None:
        assert isinstance(filename, str), \
            f"filename must be a string or None. Now its type is {type(filename)}."
    if rename_dict is not None:
        assert isinstance(rename_dict, dict), \
            f"rename_dict must be a dictionary or None. Now its type is {type(rename_dict)}."
        assert all([isinstance(k, str) and isinstance(v, str) for k, v in rename_dict.items()]), \
            "Each key and value in rename_dict (if not None) must be a string."
    metric_columns = [prefix + sep + metric for prefix in prefixes]
    for metric_col in metric_columns:
        assert metric_col in df.columns, \
            f"{metric_col} was not found as a column in df."
        try:
            df[metric_col] = df[metric_col].astype('float')
        except:
            raise TypeError(f"{metric_col} must be a numeric column.")

    # Subset the input data
    df = df[[var_column] + metric_columns].copy()

    # Create plot
    plt.figure(figsize=(12, 8))
    markers_list = ["o", "^", "s", "P", "x", "*"]
    metric_name = metric if rename_dict is None or metric not in rename_dict.keys() else rename_dict[metric]
    var_name = var_column if rename_dict is None or var_column not in rename_dict.keys() else rename_dict[var_column]
    for i, prefix in enumerate(prefixes):
        metric_col = prefix + sep + metric
        prefix_name = prefix if rename_dict is None or prefix not in rename_dict.keys() else rename_dict[prefix]
        plt.plot(df[var_column], df[metric_col], marker=markers_list[i], label=f"{metric_name} ({prefix_name})")
    plt.xlabel(var_name), plt.ylabel(metric_name)
    plt.xticks(df[var_column].values)
    plt.grid(alpha=1)
    plt.legend(loc='best')
    plt.autoscale()
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(f'{filename}.png')
    plt.show()

########################################################################################################################
# Define the plotting function for a model's AUROC
########################################################################################################################

def plot_AUROC(TPR: list[float],
               FPR: list[float],
               title: Optional[str] = None,
               filename: Optional[str] = None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve from its True Positive Rates (TPR) and False Positive Rates
    (FPR).
    :param TPR: A list of floats.
           True positive rates.
    :param FPR: A list of floats.
           Fase positive rates.
    :param title: A string or None.
           The title of the plot if a string is provided, and no title provided if None.
           Default setting: title=None
    :param filename: A string or None. If not None, a PNG file with the specified filename will be created in the
           current working directory.
           Default setting: filename=None
    :return:
    (a) A plot displayed to the IDE.
    If filename is not None:
    (b) A file named {filename}.png saved to the current working directory.
    """

    # Type and value check
    try:
        TPR = np.array(TPR)
    except:
        raise TypeError(f'TPR must be (convertible to) a numpy array. Now its type is {type(TPR)}')
    try:
        FPR = np.array(FPR)
    except:
        raise TypeError(f'FPR must be (convertible to) a numpy array. Now its type is {type(FPR)}')
    assert len(TPR.shape) ==  1, \
        f"TPR must be 1-dimensional. Now its dimension is {len(TPR.shape)}."
    assert len(FPR.shape) ==  1, \
        f"FPR must be 1-dimensional. Now its dimension is {len(FPR.shape)}."
    if title is not None:
        assert isinstance(title, str), \
            f"title must be a string or None. Now its type is {type(title)}."
    if filename is not None:
        assert isinstance(filename, str), \
            f"filename must be a string or None. Now its type is {type(filename)}."

    # Create plot
    plt.figure()
    plt.plot(FPR, TPR, color='tab:orange', lw=2, label=f'ROC curve (area = {auc(FPR, TPR):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level (area = 0.5)')
    plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
    plt.xlim([-0.01, 1.01]), plt.ylim([-0.01, 1.01])
    plt.grid(alpha=1)
    plt.legend(loc='lower right')
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(f'{filename}.png')
    plt.show()

########################################################################################################################
# Define the plotting function for the trend curve of Mean-Absolute-SHAP values
########################################################################################################################

def plot_MASHAP_trend(attributes: np.ndarray,
                      title: Optional[str] = None,
                      filename: Optional[str] = None):
    """
    Compute Mean-Absolute-SHAP (MA-SHAP) values (across samples) of a 2-dimensional SHAP matrix and plot the decaying
    trend.
    Remark 1: For multi-class classification cases, users are expected to use a single SHAP matrix for each class.
    Remark 2: For 3-dimensional time-series SHAP matrix, slice with a specific timestamp ts (e.g., attributes[:, ts, :])
              or compute the mean-absolute values across timestamps (e.g., np.mean(np.abs(attributes), axis=1) first.
    :param attributes: A 2-dimensional numpy array.
           SHAP value matrix of dimension (sample size, number of features).
    :param title: A string or None.
           The title of the plot if a string is provided, and no title provided if None.
           Default setting: title=None
    :param filename: A string or None. If not None, a .png file with the specified filename will be created to the
           current working directory.
           Default setting: filename=None
    :return:
    (a) A plot displayed to the IDE.
    If filename is not None:
    (b) A file named {filename}.png saved to the current working directory.
    """

    # Type and value check
    try:
        attributes = np.array(attributes)
    except:
        raise TypeError(f'attributes must be (convertible to) a numpy array. Now its type is {type(attributes)}.')
    assert len(attributes.shape) == 2, \
        f'attributes must be 2-dimensional. Now its dimension is {len(attributes.shape)}.'
    if title is not None:
        assert isinstance(title, str), \
            f"title must be a string or None. Now its type is {type(title)}."
    if filename is not None:
        assert isinstance(filename, str), \
            f"filename must be a string or None. Now its type is {type(filename)}."

    # Sort attributes by MA-SHAP in descending order
    attr = np.sort(np.mean(np.abs(attributes), axis=0))[::-1]

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(attr)), attr, linewidth=2, marker='o', label='MA-SHAP values of features')
    plt.xlabel('Ordered Feature Indices'), plt.ylabel('MA-SHAP Values')
    xticks = list(range(len(attr))) if len(attr) < 20 \
        else np.round(np.arange(start=0, stop=len(attr), step=len(attr)/10)).astype(int)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.xlim(left=0, right=len(attr)-1)
    plt.legend(loc='upper right')
    plt.grid(alpha=1)
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(f'{filename}.png')
    plt.show()

########################################################################################################################
# Define the plotting function for SHAP strip plots
########################################################################################################################

def plot_SHAP_strip(attributes: np.ndarray,
                    X: np.ndarray,
                    feature_names: list[str],
                    top_n_features: Optional[int] = None,
                    max_points: Optional[int] = None,
                    title: Optional[str] = None,
                    filename: Optional[str] = None,
                    random_state: Optional[Union[int, np.random.Generator]] = None):
    """
    Create a strip plot of SHAP values colored by the feature values.
    Remark 1: For multi-class classification cases, users are expected to use a single SHAP matrix for each class.
    Remark 2: For 3-dimensional time-series data and SHAP matrices, slice with a specific timestamp ts first
              (e.g., attributes[:, ts, :] and X[:, ts, :]).
    Remark 3: Due to readability, a maximum of 20 features can be displayed.
    Remark 4: Due to readability, a maximum of 300 samples/feature can be displayed.
    :param attributes: A 2-dimensional numpy array.
           SHAP value matrix of dimension (sample size, number of features).
    :param X: A 2-dimensional numpy array.
           The feature datasets of dimension (sample size, number of features) that the model explains.
    :param feature_names: A list of strings.
           Names of the features.
    :param top_n_features: A positive integer or None.
           Only the top-n features in MA-SHAP will be displayed if not None.
           Default setting: top_n_features=None
    :param max_points: A positive integer or None.
           The maximum number of samples per feature to be displayed if not None.
           Default setting: max_points=None
    :param title: A string or None.
           The title of the plot if a string is provided, and no title provided if None.
           Default setting: title=None
    :param filename: A string or None. If not None, a .png file with the specified filename will be created to the
           current working directory.
           Default setting: filename=None
    :param random_state: An integer or a numpy RNG object.
           Controls the sampling process when sample size exceed min(max_points, 400).
    :return:
    (a) A plot displayed to the IDE.
    If filename is not None:
    (b) A file named {filename}.png saved to the current working directory.
    """

    # Type and value check
    try:
        attributes = np.array(attributes)
    except:
        raise TypeError(f'attributes must be (convertible to) a numpy array. Now its type is {type(attributes)}.')
    assert len(attributes.shape) == 2, \
        f'attributes must be 2-dimensional. Now its dimension is {len(attributes.shape)}.'
    try:
        X = np.array(X)
    except:
        TypeError(f'X must be, or convertible to be, a numpy array. Now its type is {type(X)}.')
    assert len(X.shape) == 2, \
        f'X must be either 2-dimensional. Now its dimension is {len(X.shape)}.'
    assert X.shape == attributes.shape, \
        f'attributes and X must have the same dimension.'
    assert isinstance(feature_names, list), \
        f'feature_names must be a list. Now its type is {type(feature_names)}.'
    assert all([isinstance(s, str) for s in feature_names]), \
        f'Each element in feature_names must be a string.'
    assert len(feature_names) == X.shape[-1], \
        f'The length of feature_names (={len(feature_names)} must match the number of features in X (={X.shape[-1]}).'
    if top_n_features is not None:
        assert isinstance(top_n_features, int),\
            f'top_n_features must be a positive integer or None. Now its type is {type(top_n_features)}.'
        assert attributes.shape[1] >= top_n_features >= 1, \
            f'top_n_features (if not None) must be in the range [1, attributes.shape[1]]. Now it is {top_n_features}.'
    if max_points is not None:
        assert isinstance(max_points, int), \
            f'max_points must be a positive integer or None. Now its type is {type(top_n_features)}.'
        assert attributes.shape[0] >= max_points >= 1, \
            f'max_points (if not None) must be in the range [1, attributes.shape[0]]. Now it is {max_points}.'
    if title is not None:
        assert isinstance(title, str), \
            f"title must be a string or None. Now its type is {type(title)}."
    if filename is not None:
        assert isinstance(filename, str), \
            f"filename must be a string or None. Now its type is {type(filename)}."
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)   # Rely on numpy to check if random_state is valid

    # Compute MASHAP and sort features by MASHAP in descending order
    MASHAP = pd.DataFrame(np.array([feature_names, np.mean(np.abs(attributes), axis=0)]).T, columns=['Feature', 'MASHAP'])
    MASHAP = MASHAP.sort_values(by='MASHAP', axis=0, ascending=False, key=pd.to_numeric, ignore_index=True)
    sorted_feat = MASHAP['Feature'].values
    sorted_feat_idx = [feature_names.index(feat) for feat in sorted_feat]

    # Subset features to be displayed
    if top_n_features is None:
        if attributes.shape[1] <= 20:
            top_n_features = attributes.shape[1]
        else:
            warnings.warn(
                'Due to readability, showing only the top 20 features in MA-SHAP across samples.')
            top_n_features = 20
    elif top_n_features > 20:
        warnings.warn(
                'Due to readability, showing only the top 20 features in MA-SHAP across samples.')
        top_n_features = 20
    sorted_feat, sorted_feat_idx = sorted_feat[:top_n_features], sorted_feat_idx[:top_n_features]
    attributes, X = attributes[:, sorted_feat_idx], X[:, sorted_feat_idx]

    # Subset samples to be displayed
    if max_points is None:
        if attributes.shape[0] > 300:
            warnings.warn('Due to readability, showing only the 300 random samples.')
            samples = rng.integers(low=0, high=X.shape[0], size=300)
        else:
            samples = range(attributes.shape[0])
    else:
        if max_points > 300:
            warnings.warn('Due to readability, showing only the 300 random samples.')
            samples = rng.integers(low=0, high=X.shape[0], size=300)
        else:
            samples = rng.integers(low=0, high=X.shape[0], size=max_points)
    attributes, X = attributes[samples, :], X[samples, :]

    X = StandardScaler().fit_transform(X)   # A more interpretable scale of feature values
    shap_df, feature_df = pd.DataFrame(attributes, columns=sorted_feat), pd.DataFrame(X, columns=sorted_feat)
    shap_long = shap_df.melt(var_name='Feature', value_name='SHAP Value')
    feature_long = feature_df.melt(var_name='Feature', value_name='Feature Value')
    combined_df = pd.concat([shap_long, feature_long['Feature Value']], axis=1)

    # Create plot
    plt.figure(figsize=(12, 8))
    norm = plt.Normalize(combined_df['Feature Value'].min(), combined_df['Feature Value'].max())
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'white', 'red'])
    cmap = plt.get_cmap(custom_cmap)
    colors = list(cmap(norm(combined_df['Feature Value'].unique())))
    ax = sns.stripplot(data=combined_df, y='Feature', x='SHAP Value', hue='Feature Value', palette=colors,
                       jitter=0.3, alpha=0.7, linewidth=0.5, legend=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Standardized Feature Value')
    plt.xlabel('SHAP Value'), plt.ylabel('Feature (ordered by MA-SHAP)')
    plt.grid(True, axis='x', linestyle='--', alpha=1)
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(f'{filename}.png')
    plt.show()

########################################################################################################################
# Define the plotting function for features' MASHAP trend over timestamps
########################################################################################################################

def plot_MASHAP_feat_trend(attributes: np.ndarray,
                           feature_names: list[str],
                           top_n_features: Optional[int] = None,
                           y_log: bool = True,
                           title: Optional[str] = None,
                           filename: Optional[str] = None):
    """
    Create a trend plot of Mean-Absolute-SHAP (MA-SHAP) values (across samples) of features over timestamps.
    Remark 1: For multi-class classification cases, users are expected to use a single SHAP matrix for each class.
    Remark 2: Due to readability, a maximum of 20 features can be displayed.
    :param attributes: A 3-dimensional numpy array.
           SHAP values of dimension (sample size, number of timestamps, number of features).
    :param feature_names: A list of strings.
           Names of the features.
    :param top_n_features: A positive integer or None.
           Only the top-n features in MMA-SHAP will be displayed if not None.
           Default setting: top_n_features=None
    :param y_log: A boolean.
           The y-axis for MA-SHAP values will be log-scaled if True.
           Default setting: y_log=True
    :param title: A string or None.
           The title of the plot if a string is provided, and no title provided if None.
           Default setting: title=None
    :param filename: A string or None. If not None, a .png file with the specified filename will be created to the
           current working directory.
           Default setting: filename=None
    :return:
    (a) A plot displayed to the IDE.
    If filename is not None:
    (b) A file named {filename}.png saved to the current working directory.
    """

    # Type and value check
    try:
        attributes = np.array(attributes)
    except:
        raise TypeError(f'attributes must be (convertible to) a numpy array. Now its type is {type(attributes)}.')
    assert len(attributes.shape) == 3, \
        f'attributes must be 3-dimensional. Now its dimension is {len(attributes.shape)}.'
    assert isinstance(feature_names, list), \
        f'feature_names must be a list. Now its type is {type(feature_names)}.'
    assert all([isinstance(s, str) for s in feature_names]), \
        f'Each element in feature_names must be a string.'
    if top_n_features is not None:
        assert isinstance(top_n_features, int),\
            f'top_n_features must be a positive integer or None. Now its type is {type(top_n_features)}.'
        assert attributes.shape[2] >= top_n_features >= 1, \
            f'top_n_features (if not None) must be in the range [1, attributes.shape[2]]. Now it is {top_n_features}.'
    assert isinstance(y_log, bool), \
        f'y_log must be a boolean. Now its type is {type(y_log)}.'
    if title is not None:
        assert isinstance(title, str), \
            f"title must be a string or None. Now its type is {type(title)}."
    if filename is not None:
        assert isinstance(filename, str), \
            f"filename must be a string or None. Now its type is {type(filename)}."

    # Compute MASHAP and sort features by MMASHAP in descending order
    MASHAP = np.mean(np.abs(attributes), axis=0)
    MASHAP = pd.DataFrame(MASHAP, columns=feature_names)
    MASHAP = MASHAP.reindex(MASHAP.mean().sort_values(ascending=False, key=pd.to_numeric).index, axis=1)

    if top_n_features is None:
        if attributes.shape[2] <= 20:
            top_n_features = attributes.shape[2]
        else:
            warnings.warn(
                'Due to readability, showing only the top 20 features in MMA-SHAP across samples and timestamps.')
            top_n_features = 20
    elif top_n_features > 20:
        warnings.warn(
                'Due to readability, showing only the top 20 features in MMA-SHAP across samples and timestamps.')
        top_n_features = 20
    MASHAP = MASHAP[MASHAP.columns[:top_n_features]] if len(feature_names) > top_n_features else MASHAP

    colormap = get_cmap('tab20')
    plt.figure(figsize=(10, 10))
    x_data = range(MASHAP.shape[0])
    for i in range(MASHAP.shape[1]):
        y_data = MASHAP[MASHAP.columns[i]]
        if MASHAP.shape[1] <= 10:
            plt.plot(x_data, y_data, marker='.', label=MASHAP.columns[i])
        else:
            plt.plot(x_data, y_data, marker='.', label=MASHAP.columns[i],
                     color=colormap(i % MASHAP.shape[1]))
    plt.xticks(x_data)
    plt.xlim(left=min(x_data)-0.2, right=max(x_data)+0.2)
    plt.xlabel('Timestamp Indices')
    plt.ylabel('MA-SHAP Values (log-scaled)' if y_log else 'MA-SHAP Values')
    if y_log:
        plt.yscale('log')
    plt.legend(loc='best', ncols=2)
    plt.grid(alpha=1)
    if title is not None:
        plt.title(title)
    plt.title(title)
    if filename is not None:
        plt.savefig(f'{filename}.png')
    plt.show()

########################################################################################################################
# Define a function for SHAP heatmap of a feature
########################################################################################################################

def plot_SHAP_feat_heatmap(attributes: np.ndarray,
                           feature_idx: int,
                           scale_intensity: bool = True,
                           title: Optional[str] = None,
                           filename: Optional[str] = None):
    """
    Create a heatmap of a feature SHAP values with x-axis as timestamps and y-axis as samples.
    Remark 1: For multi-class classification cases, users are expected to use a single SHAP matrix for each class.
    :param attributes: A 3-dimensional numpy array.
           SHAP values of dimension (sample size, number of timestamps, number of features).
    :param feature_idx: An integer.
           The index of the feature of interest in attributes.
    :param scale_intensity: A boolean.
           Adjust the color mapping in a feature's heatmap with reference to the other features' SHAP values if True.
           Default setting: scale_intensity=True
    :param title: A string or None.
           The title of the plot if a string is provided, and no title provided if None.
           Default setting: title=None
    :param filename: A string or None. If not None, a .png file with the specified filename will be created to the
           current working directory.
           Default setting: filename=None
    :return:
    (a) A plot displayed to the IDE.
    If filename is not None:
    (b) A file named {filename}.png saved to the current working directory.
    """

    # Type and value check
    try:
        attributes = np.array(attributes)
    except:
        raise TypeError(f'attributes must be (convertible to) a numpy array. Now its type is {type(attributes)}.')
    assert len(attributes.shape) == 3, \
        f'attributes must be 3-dimensional. Now its dimension is {len(attributes.shape)}.'
    assert isinstance(feature_idx, int), \
        f'feature_idx must be an integer. Now its type is {type(feature_idx)}.'
    try:
        data = attributes[:, :, feature_idx]
    except:
        raise IndexError(f'feature_idx is not found in the third dimension of attributes.')
    assert isinstance(scale_intensity, bool), \
        f'scale_intensity must be a boolean. Now its type is {type(scale_intensity)}.'
    if title is not None:
        assert isinstance(title, str), \
            f"title must be a string or None. Now its type is {type(title)}."
    if filename is not None:
        assert isinstance(filename, str), \
            f"filename must be a string or None. Now its type is {type(filename)}."

    # Create plot
    plt.figure(figsize=(10, 10))
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'white', 'red'])
    if scale_intensity:
        plt.imshow(data, cmap=custom_cmap, aspect='auto', interpolation='nearest',
                   vmin=np.min(attributes), vmax=np.max(attributes))           # Set up value bounds if scale_intensity
    else:
        plt.imshow(data, cmap=custom_cmap, aspect='auto', interpolation='nearest')
    plt.colorbar(label='SHAP Values')
    plt.xlabel('Timestamp Indices'), plt.ylabel('Sample Indices')
    plt.xticks(range(data.shape[1]), labels=range(data.shape[1]))
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(f'{filename}.png')
    plt.show()

########################################################################################################################
# Define a plotting function for stacked SHAP bar chart
########################################################################################################################

def plot_MASHAP_bar(attributes: list[np.ndarray],
                    feature_names: list[str],
                    top_n_features: Optional[int] = None,
                    stack: bool = True,
                    title: Optional[str] = None,
                    filename: Optional[str] = None):
    """
    Create a stacked bar chart of Mean-Absolute-SHAP (MA-SHAP) values.
    Remark 1: For multi-class classification cases, users are expected to use multiple SHAP matrices in the input
              attributes, one for each class.
    Remark 2: For each 3-dimensional time-series SHAP matrix attr, slice with a specific timestamp ts
              (e.g., attr[:, ts, :]) or compute the mean-absolute values across timestamps
              (e.g., np.mean(np.abs(attr), axis=1) first.
    Remark 3: Due to readability, a maximum of 20 features can be displayed.
    :param attributes: A list of 2-dimensional numpy arrays or a single 2-dimensional numpy array.
           The list of SHAP values matrices (one for each class) or a single SHAP values matrix.
    :param feature_names: A list of strings.
           Names of features.
    :param top_n_features: A positive integer or None.
           Only the top-n features in MMA-SHAP (across samples and classes) will be displayed if not None.
           Default setting: top_n_features=None
    :param stack: A boolean.
           Bars will be stacked for different classes. Only used when a single SHAP matrix in attributes.
           Default setting: stack=True
    :param y_log: A boolean.
           The y-axis for MA-SHAP values will be log-scaled if True.
           Default setting: y_log=True
    :param title: A string or None.
           The title of the plot if a string is provided, and no title provided if None.
           Default setting: title=None
    :param filename: A string or None. If not None, a .png file with the specified filename will be created to the
           current working directory.
           Default setting: filename=None
    :return:
    (a) A plot displayed to the IDE.
    If filename is not None:
    (b) A file named {filename}.png saved to the current working directory.
    """

    # Type and value check
    if type(attributes) != list:
        attributes = [attributes, attributes] if stack else [attributes]
    elif len(attributes) == 1 and stack:
        attributes = [attributes[0], attributes[0]]
    first_attr = attributes[0]
    for idx in range(len(attributes)):
        try:
            attributes[idx] = np.array(attributes[idx])
        except:
            TypeError(f'attributes (or each element in attributes) must be (convertible to) a numpy array.')
        assert len(attributes[idx].shape) == 2, \
            f'attributes (or each element in attributes) must be 2-dimensional.'
        assert attributes[idx].shape == first_attr.shape, \
            f'Each element in attributes must have the same dimension.'
    assert isinstance(feature_names, list), \
        f'feature_names must be a list. Now its type is {type(feature_names)}.'
    assert all([isinstance(s, str) for s in feature_names]), \
        f'Each element in feature_names must be a string.'
    assert len(feature_names) == attributes[0].shape[-1], \
        (f'The length of feature_names (={len(feature_names)} must match the number of features '
         f'in attributes (={attributes[0].shape[-1]}).')
    if top_n_features is not None:
        assert isinstance(top_n_features, int),\
            f'top_n_features must be a positive integer or None. Now its type is {type(top_n_features)}.'
        assert attributes[0].shape[1] >= top_n_features >= 1, \
            f'top_n_features (if not None) must be in the range [1, attributes[0].shape[1]]. Now it is {top_n_features}.'
    assert isinstance(stack, bool), \
        f'stack must be a boolean. Now its type is {type(stack)}.'
    if title is not None:
        assert isinstance(title, str), \
            f"title must be a string or None. Now its type is {type(title)}."
    if filename is not None:
        assert isinstance(filename, str), \
            f"filename must be a string or None. Now its type is {type(filename)}."

    # Compute MASHAP by class and sort features by MMASHAP (by classes) in descending order
    MASHAP_class = [np.mean(np.abs(attr), axis=0) for attr in attributes]
    MASHAP_class = pd.DataFrame(np.array(MASHAP_class), columns=feature_names)
    MASHAP_class = MASHAP_class.reindex(MASHAP_class.mean(axis=0).
                                          sort_values(ascending=False, key=pd.to_numeric).index, axis=1)

    if top_n_features is None:
        if attributes[0].shape[1] <= 20:
            top_n_features = attributes[0].shape[1]
        else:
            warnings.warn(
                'Due to readability, showing only the top 20 features in MMA-SHAP across samples and timestamps.')
            top_n_features = 20
    elif top_n_features > 20:
        warnings.warn(
            'Due to readability, showing only the top 20 features in MMA-SHAP across samples and timestamps.')
        top_n_features = 20

    if len(feature_names) > top_n_features:
        MASHAP_class = MASHAP_class[MASHAP_class.columns[:top_n_features]]
    MASHAP_class = MASHAP_class[MASHAP_class.columns[::-1]]
    data_full = np.array(MASHAP_class.values)

    # Create plot
    plt.figure(figsize=(15, 10))
    plt.barh(MASHAP_class.columns, data_full[0, :], label=f'Class 0')    # Start with the first bar
    idx, prior_data = 1, data_full[0, :]                                 # While-loop for any other bars
    while idx < data_full.shape[0]:
        current_data = data_full[idx, :]
        plt.barh(MASHAP_class.columns, current_data, left=prior_data, label=f'Class {idx}')
        prior_data += current_data
        idx += 1
    plt.xlabel('MA-SHAP Values'), plt.ylabel('Features')
    plt.grid(alpha=1)
    if title is not None:
        plt.title(title)
    plt.legend(loc='lower right')
    if filename is not None:
        plt.savefig(f'{filename}.png')
    plt.show()

########################################################################################################################
# Define a plotting function for 3D SHAP images and movies
########################################################################################################################

def plot_SHAP_3D_movie(attributes: np.ndarray,
                       feature_names: list[str],
                       top_n_features: Optional[int] = None,
                       title: Optional[str] = None,
                       filename: Optional[str] = None):
    """
    Create a 3D movie of features' SHAP values over timestamps.
    Remark 1: For multi-class classification cases, users are expected to use multiple SHAP matrices in the input
              attributes, one for each class.
    Remark 2: Due to readability, a maximum of 20 features can be displayed (with their names) when top_n_features is
              not None.
    :param attributes: A 3-dimensional numpy array.
           SHAP values of dimension (sample size, number of timestamps, number of features).
    :param feature_names: A list of strings.
           Name of features. Used only when top_n_features is not None or the number of features is at most 20.
    :param top_n_features: A positive integer or None.
           Only the top-n features in MMA-SHAP (across samples and timestamps) will be displayed if not None.
           Default setting: top_n_features=None
    :param title: A string or None.
           The title of the plot if a string is provided, and no title provided if None.
           Default setting title=None
    :param filename: A string or None. If not None, a .png file and a HTML file with the specified filename will be
           created to the current working directory.
           Default setting: filename=None
    :return:
    (a) An interactive 3D-plot displayed in a pop-up web browser.
    If filename is not None:
    (b) A directory named {filename} is created in the current working directory containing:
        (b1) The interactive 3D-plot; path = {filename}/{filename}.HTML;
        (b2) The static image for each timestamp t; path = {filename}/{filename}_{t}.PNG;
        (b3) The animated image generated from (b1); path {filename}/{filename}_animated.PNG.
    """

    # Type and value check
    try:
        attributes = np.array(attributes)
    except TypeError:
        TypeError(f'attributes must be (convertible to) a numpy array. Now its type is {type(attributes)}.')
    assert len(attributes.shape) == 3, \
        f'attributes must be 3-dimensional. Now its dimension is {len(attributes.shape)}.'
    assert isinstance(feature_names, list), \
        f'feature_names must be a list. Now its type is {type(feature_names)}.'
    assert all([isinstance(s, str) for s in feature_names]), \
        f'Each element in feature_names must be a string.'
    if top_n_features is not None:
        assert isinstance(top_n_features, int), \
            f'top_n_features must be a positive integer or None. Now its type is {type(top_n_features)}.'
        assert attributes.shape[2] >= top_n_features >= 1, \
            f'top_n_features (if not None) must be in the range [1, attributes.shape[2]]. Now it is {top_n_features}.'
    if title is not None:
        assert isinstance(title, str), \
            f"title must be a string or None. Now its type is {type(title)}."
    if filename is not None:
        assert isinstance(filename, str), \
            f"filename must be a string or None. Now its type is {type(filename)}."

    # Compute MASHAP and sort features by MMASHAP in descending order
    MASHAP = pd.DataFrame(np.mean(np.abs(attributes), axis=1), columns=range(attributes.shape[2]))
    MASHAP = MASHAP.reindex(MASHAP.mean(axis=0).sort_values(ascending=False, key=pd.to_numeric).index, axis=1)

    show_feat_name = True
    if top_n_features is None:
        if attributes.shape[2] <= 20:
            top_n_features = attributes.shape[2]
        else:
            show_feat_name = False
    else:
        if top_n_features > 20:
            warnings.warn('Due to readability, showing only the top 20 features in MMA-SHAP across samples '
                          'and timestamps.')
        top_n_features = min(20, top_n_features)

    if not show_feat_name:
        y_ticks = [0, attributes.shape[2]-1]
        y_tick_labels = y_ticks
        y_label = 'Ordered Feature Indices'
    else:
        attributes = attributes[:, :, MASHAP.columns[:top_n_features]]
        y_ticks = list(range(top_n_features))
        y_tick_labels = [feature_names[i] for i in MASHAP.columns[:top_n_features]]
        y_label = 'Features'
    x_label, z_label = 'Sample Indices', 'SHAP Values'
    z_min, z_max = attributes.min(), attributes.max()

    # Create a directory for the outputs (if required)
    if filename is not None:
        os.makedirs(filename, exist_ok=True)

    # Layout of the static images
    layout_dict_PNG = dict(xaxis=dict(title=dict(text=x_label, font=dict(size=8)),
                                      tickfont=dict(size=8)),
                           yaxis=dict(title=dict(text=y_label, font=dict(size=8)),
                                      tickmode='array',
                                      tickvals=y_ticks,
                                      ticktext=y_tick_labels,
                                      title_font=dict(size=8), tickfont=dict(size=8)),
                           zaxis=dict(title=dict(text=z_label, font=dict(size=8)),
                                      title_font=dict(size=8), tickfont=dict(size=8),
                                      range=[attributes.min(), attributes.max()]),
                           camera=dict(eye=dict(x=2, y=2, z=1)),
                           aspectmode='manual',
                           aspectratio=dict(x=1, y=1, z=0.8))

    # Layout of the 3D animation
    layout_dict_HTML = dict(xaxis=dict(title=dict(text=x_label, font=dict(size=12)),
                                       tickfont=dict(size=12)),
                            yaxis=dict(title=dict(text=y_label, font=dict(size=12)),
                                       tickmode='array',
                                       tickvals=y_ticks,
                                       ticktext=y_tick_labels,
                                       title_font=dict(size=12), tickfont=dict(size=12)),
                            zaxis=dict(title=dict(text=z_label, font=dict(size=12)),
                                       title_font=dict(size=12), tickfont=dict(size=12),
                                       range=[attributes.min(), attributes.max()]),
                            camera=dict(eye=dict(x=2, y=2, z=1)))

    fig = go.Figure()   # Base object of the 3D animation
    frames = []         # A list storing each frame of the 3D animation
    PNG_files = []      # A list storing each static image file (if required)
    for t in range(attributes.shape[1]):
        data_slice = attributes[:, t, :]    # Slice the data to a given timestamp
        samples, features = np.arange(data_slice.shape[0]), np.arange(data_slice.shape[1])
        X, Y = np.meshgrid(samples, features)
        go_surface = go.Surface(z=data_slice.T, x=X, y=Y, colorscale='viridis', cmin=z_min, cmax=z_max,
                                colorbar=dict(title=dict(text=z_label, font=dict(size=13)),
                                              tickfont=dict(size=12), len=0.5, x=0.9))
        if t == 0:
            fig.add_trace(go_surface)   # base frame of the 3D animation
        frames.append(go.Frame(data=[go_surface], name=f"Timestamp {t}"))   # add frame for the 3D animation
        if filename is not None:
            go_surface.colorbar = dict(title=dict(text=z_label, font=dict(size=8)), tickfont=dict(size=6),
                                       len=0.6, x=0.85)
            frame_fig = go.Figure(data=[go_surface])
            title_str = f'{title} (Timestamp {t})' if title is not None else f'Timestamp {t}'
            frame_fig.update_layout(scene=layout_dict_PNG,
                                    title=dict(text=title_str, font=dict(size=10), x=0.2, y=0.8),
                                    margin=dict(l=0, r=0, t=0, b=0))
            # Create static image PNG file
            frame_fig.write_image(f"{filename}/{filename}_{t}.png", width=1080, height=1080, scale=2)
            with Image.open(f"{filename}/{filename}_{t}.png") as img:
                img = img.crop((300, 360, 2000, 1840))
                img.save(f"{filename}/{filename}_{t}.png")      # Overwrite each PNG file with its cropped version
                PNG_files.append(img)
    fig.frames = frames

    # Button settings in the 3D animation
    buttons = [{"buttons":
                [{"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                  "label": "Play",
                  "method": "animate"},
                 {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                  "label": "Pause",
                  "method": "animate"}],
                 "direction": "left",
                 "pad": {"r": 10, "t": 87},
                 "showactive": False,
                 "type": "buttons",
                 "x": 0.2,
                 "xanchor": "right",
                 "y": 0.2,
                 "yanchor": "top",
                }]

    # Slider settings in the 3D animation
    sliders = [{"steps":
                [{"args": [[f"Timestamp {t}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                  "label": f"{t}",
                  "method": "animate"} for t in range(attributes.shape[1])],
                 "transition": {"duration": 0},
                 "x": 0.2,
                 "y": 0.1,
                 "currentvalue": {"font": {"size": 20},
                                 "prefix": "Timestamp Index: ",
                                 "visible": True,
                                 "xanchor": "center"},
                "len": 0.7,
               }]

    # Configure the layout
    fig.update_layout(scene=layout_dict_HTML, updatemenus=buttons, sliders=sliders, margin=dict(l=0, r=0, t=0, b=0))
    if title is not None:
        fig.update_layout(title=dict(text=f'{title}', font=dict(size=16), x=0.3, y=0.9))
    if filename is not None:
        PNG_files[0].save(f"{filename}/{filename}_animated.png", save_all=True, append_images=PNG_files[1:],
                          duration=500, loop=0)             # Create animated image PNG file
        fig.write_html(f'{filename}/{filename}.html')       # Create interactive HTML file
    fig.show()

########################################################################################################################
