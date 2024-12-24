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
import torch
import warnings
from imblearn.metrics import specificity_score                  # For classification
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)                         # For classification
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             root_mean_squared_error, r2_score) # For regression
from sklearn.preprocessing import label_binarize                # For classification
from typing import Literal, Optional

########################################################################################################################
# Standard classification metrics (Accuracy, F1, Precision, Recall, Specificity)
########################################################################################################################


def classify_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     prefix: str = "",
                     average: Optional[Literal['binary', 'micro', 'macro', 'weighted']] = 'weighted',
                     return_cm: bool = False):
    """
    Compute 5 metrics for binary/multi-class classification tasks: Accuracy, F1, Precision, Recall, and Specificity
    See https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification

    :param y_true: A one-dimensional numpy array.
           Values of the true label.
    :param y_pred: A one-dimensional numpy array.
           Values of the predicted label.
    :param prefix: A string.
           The prefix (e.g. 'Train_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix=""
    :param average: A string in ['binary', 'micro', 'macro', 'weighted'] or None.
           Configuration for averaging techniques.
           'binary': use only for binary classification
           'micro': compute metric globally by counting the total true positives, false negatives, and false positives
           'macro': compute the averaged metric for each label in an unweighted manner
           'weighted': similar to 'macro' but weighted by the number of true instances for each label
           None: compute metric for each label without averaging
           Default setting: average='weighted'
    :param return_cm: A boolean.
           Return confusion matrix as well if True.
           Default setting: return_cm=False
    :return:
    (a) A dictionary with keys as {prefix}{metric name} and values as results.
    If return_cm=True:
    (b) The confusion matrix as a numpy.ndarray.
    """

    # Type and value check
    try:
        y_true = np.array(y_true)
    except:
        raise TypeError(f"y_true must be (convertible to) a numpy array. Now its type is {type(y_true)}.")
    try:
        y_pred = np.array(y_pred)
    except:
        raise TypeError(f"y_pred must be (convertible to) a numpy array. Now its type is {type(y_pred)}.")
    assert len(y_true.shape) == 1, \
        f"The true labels y_true must be one-dimensional. Now it is {len(y_true.shape)}-dimensional."
    assert len(y_pred.shape) == 1, \
        f"The predicted labels y_pred must be one-dimensional. Now it is {len(y_pred.shape)}-dimensional."
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."
    if average is not None:
        assert average in ['binary', 'micro', 'macro', 'weighted'], \
            (f"average (if not None) must be a string in ['binary', 'micro', 'macro', 'weighted']. "
             f" Now it is {average}.")
    assert isinstance(return_cm, bool), \
        f'return_cm must be a boolean. Now its type is {type(return_cm)}.'
    n_classes = len(np.unique(y_true))
    n_classes_pred = len(np.unique(y_pred))
    assert n_classes > 1, \
        f"The number of classes for the true label must be greater than 1. Now it is {n_classes}"
    assert n_classes_pred <= n_classes, \
        (f"The number of classes for the true label (={n_classes}) must be greater or equal to the number of classes "
         f"for the predicted label (={n_classes_pred}).")
    if n_classes == 2:
        average = 'binary'
    result = {}     # Storing the computed statistics

    # Metric 1: Accuracy
    # - Accuracy works the same for binary and multiclass comparison.
    # - No zero division handling needed because of the assertion that the number of classes in the true label > 1.
    # - Returning a float object.
    acc = accuracy_score(y_true, y_pred)
    result[f'{prefix}Accuracy'] = acc

    # Metric 2: Specificity
    # - No zero division handling needed because of the assertion that the number of classes in the true label > 1.
    # - Returning a float object if average in ['macro', 'micro', 'weighted'], or a list if average == None.
    spec = specificity_score(y_true, y_pred, average=average)
    spec = list(spec) if isinstance(spec, np.ndarray) else spec
    result[f'{prefix}Specificity'] = spec

    # Metric 3, 4, 5: F1, Precision, Recall
    # - Cases of zero division is represented by numpy.nan with a warning.
    # - Returning a float object if average in ['macro', 'micro', 'weighted'], or a list if average == None.
    for (metric_str, metric_func) in [('Precision', precision_score),
                                      ('Recall', recall_score),
                                      ('F1', f1_score)]:
        score = metric_func(y_true, y_pred, average=average, zero_division=np.nan)
        warn_msg = f"Division by zero error occurred when computing {metric_str.lower()}. Returning numpy.nan."
        if type(score) is np.ndarray:
            score = list(score)
            if np.any(np.isnan(score)):
                warnings.warn(warn_msg)
        elif np.isnan(score):
            warnings.warn(warn_msg)
        result[f'{prefix}{metric_str}'] = score

    result = {k: result[k] for k in sorted(result.keys())}
    if return_cm:
        return result, confusion_matrix(y_true, y_pred)
    else:
        return result

########################################################################################################################
# Classification metric (with predicted probability measures): AUROC
########################################################################################################################


def classify_AUROC(y_true: np.ndarray,
                   y_score: np.ndarray,
                   prefix: str = "",
                   average: Optional[Literal['micro', 'macro']] = 'micro',
                   auroc_only: str = True):
    """
    Compute AUROC (Area Under the Receiver Operating Characteristic curve), TPR (True Positive Rates), and FPR (False
    Positive Rates) by comparing the true labels and the predicted probability measures (NOT the predicted labels). In
    the multi-class classification case, this function adopts the one-versus-rest strategy (comparing each class with
    the rest of the classes) using either of the following averaging approaches:
    (i) micro-averaging: focus on overall performance by treating each sample equally;
    (ii) macro-averaging: focus on per-class performance by treating each class equally important.

    :param y_true: A one-dimensional numpy array.
           Values of the true label.
    :param y_score: A one-dimensional (binary) or two-dimensional (multi-class) numpy array.
           Probability measures of each class.
    :param prefix: A string.
           The prefix (e.g. 'Train_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix=""
    :param average: A string in ['micro', 'macro'] or None.
           Averaging approach for multi-class cases.
           Default setting: average='micro'
    :param auroc_only: A boolean.
           Return AUROC only if True, and both the lists of TPR and FPR as well if False.
           Default setting: auroc_only=True
    :return:
    (a) A dictionary with key as {prefix}AUROC and value as the AUROC value.
    If auroc_only=False:
    (b) A list of TPR values.
    (c) A list of FPR values.
    """
    try:
        y_true = np.array(y_true)
    except:
        raise TypeError(f"y_true must be (convertible to) a numpy array. Now its type is {type(y_true)}.")
    try:
        y_score = np.array(y_score)
    except:
        raise TypeError(f"y_score must be (convertible to) a numpy array. Now its type is {type(y_score)}.")
    assert len(y_true.shape) == 1, \
        f"y_true must be one-dimensional. Now it is {len(y_true.shape)}-dimensional."
    assert len(y_score.shape) in [1, 2], \
        f"y_score must be one-/two-dimensional. Now it is {len(y_score.shape)}-dimensional."
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."
    if average is not None:
        assert average in ['micro', 'macro'], \
            f"average (if not None) must be a string in ['micro', 'macro']. Now it is {average}."
    assert isinstance(auroc_only, bool), \
        f'auroc_only must be a boolean. Now its type is {type(auroc_only)}.'
    n_classes = len(np.unique(y_true))
    assert n_classes > 1, \
        f"The number of classes for the true label must be greater than 1. Now it is {n_classes}"

    # AUROC in the binary case
    if n_classes == 2:
        assert y_score.shape == y_true.shape, \
            f"The probability measures y_score should be in the same shape as the true labels y_true"
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)

    # AUROC in the multi-class case
    else:
        assert len(y_score.shape) == 2, \
            (f"The probability measures y_score must be two-dimensional in the non-binary case. "
             f"Now it is {len(y_score.shape)}-dimensional.")

        y_true_binarized = label_binarize(y_true, classes=np.arange(n_classes))

        if average == 'micro':
            fpr, tpr, _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())
            auroc = roc_auc_score(y_true, y_score, average='micro', multi_class='ovr')
        else:
            auroc = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
            fpr_list, tpr_list = [], []
            for i in range(n_classes):
                fpr_i, tpr_i, _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
                fpr_list.append(fpr_i)
                tpr_list.append(tpr_i)
            fpr = np.linspace(0, 1, 1000)        # A relatively fine grid for interpolation
            tpr = np.zeros_like(fpr)
            for i in range(n_classes):
                tpr += np.interp(fpr, fpr_list[i], tpr_list[i])
            tpr /= n_classes

    if auroc_only:
        return {f'{prefix}AUROC': auroc}
    else:
        return {f'{prefix}AUROC': auroc}, list(tpr), list(fpr)

########################################################################################################################
# Classification metric (with predicted probability measures): AIC, BIC, and negative log-likelihood (NLL)
########################################################################################################################


def classify_AIC_BIC(y_true: np.ndarray,
                     y_score: np.ndarray,
                     n_params: int,
                     prefix: str = ""):
    """
    Compute AIC and BIC scores (and NLL) by comparing the true labels and the predicted probability measures (NOT the
    predicted labels). Notice that the number of free model parameters (n_params) is required for the calculation.

    :param y_true: A one-dimensional numpy array.
           Values of the true label.
    :param y_score: A one-dimensional (binary) or two-dimensional (multi-class) numpy array.
           Probability measures of each class.
    :param n_params: A positive integer.
           Number of free model parameters.
    :param prefix: A string.
           The prefix (e.g. 'Train_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix=""
    :return:
    A dictionary with keys as {prefix}{metric name} and values as results.
    """

    # Type and value check
    try:
        y_true = np.array(y_true)
    except:
        raise TypeError(f"y_true must be (convertible to) a numpy array. Now its type is {type(y_true)}.")
    try:
        y_score = np.array(y_score)
    except:
        raise TypeError(f"y_score must be (convertible to) a numpy array. Now its type is {type(y_score)}.")
    assert len(y_true.shape) == 1, \
        f"y_true must be one-dimensional. Now it is {len(y_true.shape)}-dimensional."
    assert len(y_score.shape) in [1, 2], \
        f"y_score must be one-/two-dimensional. Now it is {len(y_score.shape)}-dimensional."
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."
    assert isinstance(n_params, int), \
        f'n_params must be an integer. Now its type is {type(n_params)}.'
    assert n_params >= 1, \
        f'n_params must be a positive integer. Now it is {n_params}.'
    n_classes = len(np.unique(y_true))
    assert n_classes > 1, \
        f"The number of classes for the true label must be greater than 1. Now it is {n_classes}"

    # Define sample size
    n_samples = len(y_true)

    # Remarks for calculation
    # AIC = -2 * LL + 2 * n_params, see https://en.wikipedia.org/wiki/Akaike_information_criterion
    # BIC = -2 * LL + ln(n_samples) * n_params, see https://en.wikipedia.org/wiki/Bayesian_information_criterion
    # where LL denotes the log-likelihood. In other words, we can compute AIC and BIC as follows:
    # AIC = 2 * NLL + 2 * n_params
    # BIC = 2 * NLL + ln(n_samples) * n_params
    # where NLL denotes the Negative Log-Likelihood.
    # Reference: https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81

    if n_classes == 2:
        # In the binary case, binary cross-entropy is identical to negative log-likelihood.
        NLL = torch.nn.BCELoss(reduction='mean')(torch.DoubleTensor(y_score), torch.DoubleTensor(y_true)).item()
    else:
        NLL = torch.nn.NLLLoss(reduction='mean')(torch.Tensor(y_score), torch.Tensor(y_true).long()).item()
    AIC = (2 * NLL) + (n_params * 2)
    BIC = (2 * NLL) + (n_params * np.log(n_samples))
    return {f'{prefix}AIC': AIC,
            f'{prefix}BIC': BIC,
            f'{prefix}NLL': NLL}

########################################################################################################################
# Standard regression metrics (MAE, MAPE, MSE, R2, RMSE)
########################################################################################################################


def regress_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    prefix: str = ""):
    """
    Compute 5 metrics for regression tasks.: MAE, MAPE, MSE, R2, and RMSE

    :param y_true: A one-dimensional numpy array.
           Values of the true label.
    :param y_pred: A one-dimensional numpy array.
           Values of the predicted label.
    :param prefix: A string.
           The prefix (e.g. 'Train_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix=""
    :return:
    A dictionary with keys as {prefix}{metric name} and values as results.
    """

    # Type and value check
    try:
        y_true = np.array(y_true)
    except:
        raise TypeError(f"y_true must be (convertible to) a numpy array. Now its type is {type(y_true)}.")
    try:
        y_pred = np.array(y_pred)
    except:
        raise TypeError(f"y_pred must be (convertible to) a numpy array. Now its type is {type(y_pred)}.")
    assert len(y_true.shape) == 1, \
        f"The true labels y_true must be one-dimensional. Now it is {len(y_true.shape)}-dimensional."
    assert len(y_pred.shape) == 1, \
        f"The predicted labels y_pred must be one-dimensional. Now it is {len(y_pred.shape)}-dimensional."
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."

    # Compute metrics imported from sklearn.metrics
    MSE = mean_squared_error(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = root_mean_squared_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true, y_pred)
    return {f'{prefix}MAE': MAE,
            f'{prefix}MAPE': MAPE,
            f'{prefix}MSE': MSE,
            f'{prefix}R2': R2,
            f'{prefix}RMSE': RMSE}

########################################################################################################################
# Regression metrics: AIC and BIC
########################################################################################################################


def regress_AIC_BIC(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    n_params: int,
                    prefix: str = ""):
    """
    Compute the AIC and BIC scores from the MSE of comparing the true values and their predicted labels. Notice that
    the number of free model parameters (n_params) is required for the calculation.

    :param y_true: A one-dimensional numpy array.
           Values of the true label.
    :param y_pred: A one-dimensional numpy array.
           Values of the predicted label.
    :param n_params: A positive integer.
           Number of free model parameters.
    :param prefix: A string.
           The prefix (e.g. 'Train_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix=""
    :return:
    A dictionary with keys as {prefix}{metric name} and values as results.
    """

    # Type and value check
    try:
        y_true = np.array(y_true)
    except:
        raise TypeError(f"y_true must be (convertible to) a numpy array. Now its type is {type(y_true)}.")
    try:
        y_pred = np.array(y_pred)
    except:
        raise TypeError(f"y_pred must be (convertible to) a numpy array. Now its type is {type(y_pred)}.")
    assert len(y_true.shape) == 1, \
        f"The true labels y_true must be one-dimensional. Now it is {len(y_true.shape)}-dimensional."
    assert len(y_pred.shape) == 1, \
        f"The predicted labels y_pred must be one-dimensional. Now it is {len(y_pred.shape)}-dimensional."
    assert isinstance(n_params, int), \
        f'n_params must be an integer. Now its type is {type(n_params)}.'
    assert n_params >= 1, \
        f'n_params must be a positive integer. Now it is {n_params}.'
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."

    # Define sample size
    n_samples = len(y_true)

    # Remarks for calculation
    # AIC = MSE + 2 * n_params
    # BIC = MSE + ln(n_samples) * n_params
    # Reference: https://darrenho.github.io/AMA/1_regression1.pdf

    MSE = (sum((y_pred - y_true) ** 2)) / n_samples
    AIC = MSE + (n_params * 2)
    BIC = MSE + (n_params * np.log(n_samples))
    return {f'{prefix}AIC': AIC,
            f'{prefix}BIC': BIC}

########################################################################################################################
