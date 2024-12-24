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
import torch
import warnings
from .algo import (ANN_Classifier, ANN_Regressor, LSTM_Classifier, LSTM_Regressor, Transformer_Classifier,
                   Transformer_Regressor, TCN_Classifier, TCN_Regressor, train_model, test_model,
                   LossFunction, EarlyStopping)
from .metrics import classify_metrics, classify_AUROC, classify_AIC_BIC, regress_metrics, regress_AIC_BIC
from captum.attr import GradientShap
from functools import partial
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Union, Literal, Optional

########################################################################################################################
# Define a RiskPath model class for classification
########################################################################################################################


class RPClassifier:
    """
    A PyTorch RiskPath class for classification.

    A. Runtime parameters
    ---------------------
    A1. base_model: An object of any class in the list [ANN_Classifier, LSTM_Classifier, Transformer_Classifier,
                    TCN_Classifier].
        The embedded deep-learning model to be used.
    A2. param_grid: A list.
        The list of values of the tunable parameter of the base_model in A1. Below are the corresponding parameters:
        ANN_Classifier: n_units
        LSTM_Classifier: n_units
        Transformer_Classifier: d_model
        TCN_Classifier: n_units
    A3. verbose: An integer in [0, 1].
        Verbosity. No logging if 0, and information about the grid search over A2 will be displayed if 1.
        Default setting: verbose=1
    A4. kwargs: (Any extra runtime parameters of base_model)
        Example: bidirectional=True when base_model is an LSTM_Classifier object
        Example: n_heads=16 when base_model is a Transformer_Classifier object

    B. Attributes
    -------------
    B1. metrics_list: The list of available evaluation metrics to be used.
    B2. is_fitted: A boolean indicating whether the RiskPath model has been fitted.
    B3. is_evaluated: A boolean indicating whether the RiskPath model has been evaluated.
    B4. n_features: A positive integer indicating the number of features when fitting the RiskPath model.
    B5. n_timestamps: A positive integer indicating the number timestamps when fitting the RiskPath model.
    B6. n_classes: A positive integer indicating the number of classes in the target when fitting the RiskPath model.
    B7. criterion: A loss function from torch.nn when fitting the RiskPath model.
    B8. models_dict: A dictionary with keys as numbers of units in A2 and values as the corresponding fitted
                     ANN/LSTM model.
    B9. models_performance_dict: A dictionary with keys as numbers of units in A2 and values and dictionary of
                                 performance statistics.
    B10. models_performance_df: A pandas.DataFrame with rows corresponding to the numbers of units in A2 and columns as
                                the performance statistics.
    B11. TFPR_dict: A dictionary with keys as partitions' strings (e.g. 'Train', 'Val') and values as sub-dictionaries
                    with keys and number of units in A2 and values as [list of TPR values, list of FPR values].
    B12. partial_SHAP: A dictionary with keys as the numbers of units in A2 and values as the partial functions used to
                       compute SHAP values when applying get_SHAP in CX.
    (A1-A4 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. fit(X, y, criterion, optimizer, cv, n_epochs, earlyStopper, verbose_epoch, val_metric, random_state, kwargs)
        Fit each model created according to the grid in A2 with the feature set X and target y.
        :param X: A 2- or 3-dimensional numpy array (or Torch.Tensor).
               Samples of the feature set with dimension as (sample size, number of features) or (sample size,
               number of timestamps, number of features).
        :param y: A 1-dimensional numpy array (or Torch.Tensor).
               Samples of the target with dimension as (sample size,).
        :param criterion: A loss function from torch.nn.
               See https://pytorch.org/docs/main/nn.html#loss-functions. Make sure you specify it WITH brackets.
               Example: torch.nn.NLLLoss()
        :param optimizer: An torch.optim.Optimizer object.
               See https://pytorch.org/docs/stable/optim.html. Make sure you specify it WITHOUT brackets.
               Example: torch.nn.AdamW
        :param cv: An integer.
               The number of folds used in the cross-validation splitting strategy.
               Default setting: cv=5
        :param n_epochs: A positive integer.
               The number of maximum epochs to be run in training the model.
               Default setting: n_epochs=100
        :param earlyStopper: An EarlyStopping object or None.
               It aims to speed up training time and prevent over-fitting.
               Default setting: earlyStopper=None
        :param verbose_epoch: A positive integer or None.
               If integer, it controls the frequency of printing the training and validation losses with a rate of every
               {verbose_epoch} epochs. No logging will be printed if None.
               Default setting: verbose_epoch=None
        :param val_metric: A string in B1.
               Specify the metric used to judge which model in the cross-validation process is the best.
               Default setting: val_metric='AUROC'
        :param random_state: An integer, a numpy.random.RandomState instance, or None.
               The seed of the random number generator used in the cross-validation process.
               Default setting: random_state=None
        :param kwargs: (Any extra runtime parameters of optimizer)
               Example: lr=0.001 for the learning rate parameter of the optimizer.
    C2. evaluate(X, y):
        Evaluate the models fitted in C2 with the feature set X and the target y.
        :param X: A 2- or 3-dimensional numpy array (or Torch.Tensor).
               Samples of the feature set with dimension as (sample size, number of features) or (sample size,
               number of timestamps, number of features).
        :param y: A 1-dimensional numpy array (or Torch.Tensor).
               Samples of the target with dimension as (sample size,).
    C3. get_performance()
        Obtain a pandas.DataFrame summarizing the performance statistics of each fitted (and evaluated) model.
        :return: A Pandas DataFrame with rows as models and columns as evaluation metrics.
    C4. get_best_performance(partition, metric)
        Obtain a dictionary from the row of C3 with the best user-specified metric.
        :param partition: A string in ['Train', 'Val'] or any user-specified string used in evaluation.
               Identify which partition of the dataset where the metric is evaluated at.
        :param metric: A string in B1.
               The name of the evaluation metric.
        :return: A dictionary with keys as names of characteristics or metrics of the model and values as the values of
                 those characteristics or metrics.
    C5. get model(param)
        Obtain the fitted model of the specified parameter.
        :param param: A positive integer in the param_grid in A2.
        :return: The fitted model of the user-specified number of units.
    C6. get_best_model(partition, metric)
        Obtain the best-fitting model according to the specified metric (relative to the given partition).
        :param partition: A string in ['Train', 'Val'] or any user-specified string used in evaluation.
               Identify which partition of the dataset where the metric is evaluated at.
        :param metric: metric: A string in B1.
               The name of the evaluation metric.
        :return: The fitted model of the user-specified metric.
    C7. get_TPF_FPR(param, partition)
        Obtain the two lists of true positive rates and false positive rates (relative to the given partition) of the
        model with the specified number of units.
        :param param: A positive integer in the param_grid in A2.
        :param partition: A string in ['Train', 'Val'] or any user-specified string used in evaluation.
               Identify which partition of the dataset where the metric is evaluated at.
        :return:
        (a) A list of true positive rates.
        (b) A list of false positive rates.
    C8. get_SHAP(param, X)
        Compute the SHAP values (for each class) by explaining X with the model identified by param (where the
        explanation baseline is the dataset used to train the model).
        :param param: A positive integer in the param_grid in A2.
        :param X: A 2- or 3-dimensional numpy array (or Torch.Tensor).
               Samples of the feature set with dimension as (sample size, number of features) or (sample size,
               number of timestamps, number of features).
        :return:
        When n_classes (in B6) is 2, return a 2- or 3-dimensional numpy array of SHAP values (depending on the
        dimension of X) When n_classes > 2, return a list (with length = n_classes) of 2- or 3-dimensional numpy arrays
        of SHAP values, one for each corresponding class.

    D. Remarks
    ----------
    Runtime parameter settings (in A4) of the base_model (in A1) are as follows:
    ANN_Classifier: n_layers (default: 2)
    LSTM_Classifier: n_layers (default: 2), bidirectional (default: False)
    Transformer_Classifier: n_layers (default: 2), n_units (default: 1024), n_heads (default: 8)
    TCN_Classifier: n_layers (default: 2), kernel_size (default: 3)
    """
    def __init__(self,
                 base_model: Union[ANN_Classifier, LSTM_Classifier, Transformer_Classifier, TCN_Classifier],
                 param_grid: list[int],
                 verbose: int = 1,
                 **kwargs):

        # Type and value check
        assert base_model in [LSTM_Classifier, ANN_Classifier, Transformer_Classifier, TCN_Classifier], \
            (f'model must be an object from [ANN_Classifier, LSTM_Classifier, Transformer_Classifier, TCN_Classifier]. '
             f'Now its type is {type(base_model)}.')
        self.base_model = base_model
        try:
            param_grid = list(param_grid)
        except:
            raise TypeError(f'param_grid must be (convertible to) a list. Now its type is {type(param_grid)}.')
        assert all([isinstance(param, int) and param > 0 for param in param_grid]), \
            f'All elements in param_grid must be a positive integer.'
        self.param_grid = param_grid
        assert verbose in [0, 1], \
            f"verbose must be in [0, 1]. Now its value is {verbose}."
        self.verbose = verbose
        self.kwargs = kwargs

        # Define attributes
        self.metrics_list = ['AIC', 'AUROC', 'Accuracy', 'BIC', 'F1', 'NLL', 'Precision', 'Recall', 'Specificity', 'loss']
        self.is_fitted = False              # Boolean indicating whether the model has been fitted
        self.is_evaluated = False           # Boolean indicating whether the model has been evaluated
        self.n_features = None              # Number of features to be known from the data when fitted.
        self.n_timestamps = None            # Number of timestamps to be known from the data when fitted.
        self.n_classes = None               # Number of classes to be known from the data when fitted.
        self.criterion = None               # Criterion (aka loss) for back-propagation to be input when fitted.
        self.models_dict = {}               # Store the fitted models (only best in k-fold)
        self.models_performance_dict = {}   # Store the performance of the fitted models (only best in k-fold)
        self.models_performance_df = None   # Similar to .models_performance_dict but in Pandas.DataFrame
        self.TFPR_dict = {'Train': {k: None for k in param_grid},  # Store the TPR and FPR statistics
                          'Val': {k: None for k in param_grid}}    # for each partition and each param in A2
        self.partial_SHAP = {}              # Store the partial functions for SHAP calculation

    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            criterion: LossFunction,
            optimizer: torch.optim.Optimizer,
            cv: int = 5,
            n_epochs: int = 100,
            earlyStopper: Optional[EarlyStopping] = None,
            verbose_epoch: Optional[int] = None,
            val_metric: Literal['AIC', 'AUROC', 'Accuracy', 'BIC', 'F1',
                                'NLL', 'Precision', 'Recall', 'Specificity', 'loss'] = 'AUROC',
            random_state: Optional[Union[int, np.random.RandomState]] = None,
            **kwargs):
        # Type check for X and y, n_epochs, earlyStopper, verbose_epoch will be performed in dl_base.train_model.
        # No type check for criterion and optimizer because PyTorch did not define the associated class.
        assert isinstance(cv, int), \
            f'cv has to be a positive integer. Now its type is {type(cv)}.'
        assert cv >= 1, \
            f'cv has to be a positive integer. Now it is {cv}.'
        assert val_metric in self.metrics_list, \
            f'val_metric must be in {self.metrics_list}. Now it is {val_metric}.'

        # Create indices for cross-validation (and rely on sklearn to check if random_state is valid)
        skf = StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=True)

        # Update attributes from training data
        self.n_features = X.shape[-1]
        if len(X.shape) == 3:
            self.n_timestamps = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.criterion = criterion

        # Start grid search
        for param in self.param_grid:

            # Storing characteristics of the best-performing model during cross-validation
            best_val_score, best_model, best_performance = float('-inf'), None, None
            best_train_TFPR, best_val_TFPR = None, None

            # Start cross-validation
            for k_idx, (train, val) in enumerate(skf.split(X, y), 1):
                X_train, X_val = np.take(X, train, axis=0), np.take(X, val, axis=0)
                y_train, y_val = np.take(y, train), np.take(y, val)

                # Create prediction model
                if self.base_model == ANN_Classifier:
                    M = self.base_model(n_feat=X_train.shape[-1], n_units=int(param), n_classes=self.n_classes)
                elif self.base_model == LSTM_Classifier:
                    M = self.base_model(n_feat=X_train.shape[-1], n_units=int(param), n_classes=self.n_classes,
                                        **self.kwargs)
                elif self.base_model == Transformer_Classifier:
                    M = self.base_model(n_feat=X_train.shape[-1], n_timestamps=self.n_timestamps,
                                        d_model=int(param), n_classes=self.n_classes, **self.kwargs)
                elif self.base_model == TCN_Classifier:
                    M = self.base_model(n_feat=X_train.shape[-1], n_units=int(param), n_classes=self.n_classes,
                                        **self.kwargs)
                else:
                    M = None    # Unnecessary but here for consistency

                # Configure prediction model
                M.set_device('cuda:0' if torch.cuda.is_available() else 'cpu')
                M.init_Xavier_weights()

                # Train models by k-fold validation
                if self.verbose == 1:
                    print(f"\nTraining {k_idx}/{cv} model with param={param}...", flush=True)
                fit_result = train_model(M, X_train, y_train, X_val, y_val, n_epochs,
                                         criterion, optimizer, earlyStopper, verbose_epoch, **kwargs)
                if earlyStopper is not None:
                    earlyStopper.reset()
                n_params = M.get_n_params()     # For AIC/BIC calculation
                train_tfpr_pair, val_tfpr_pair = [], []     # Temporary storage of TPR and FPR values

                # Obtain training and validation performance statistics
                for (X_, y_, prefix) in [(X_train, y_train, 'Train_'), (X_val, y_val, 'Val_')]:
                    loss, y_pred_ = test_model(M, X_, y_, criterion, prefix=prefix, return_pred=True)
                    y_pred_ = y_pred_.cpu().numpy()
                    y_pred_lab_ = np.where(y_pred_ <.5, 0, 1) if self.n_classes == 2 else np.argmax(y_pred_, axis=1)
                    fit_result |= classify_metrics(y_, y_pred_lab_, prefix=prefix)
                    auroc_, tpr_ ,fpr_ = classify_AUROC(y_, y_pred_, prefix=prefix, auroc_only=False)
                    fit_result |= auroc_
                    if prefix == 'Train_':
                        train_tfpr_pair = [tpr_, fpr_]
                    else:
                        val_tfpr_pair = [tpr_, fpr_]
                    fit_result |= classify_AIC_BIC(y_, y_pred_, n_params=n_params, prefix=prefix)

                # Select the metric to compare model performance
                score_name = f'Val_{val_metric}'
                score = -fit_result[score_name] if val_metric in ['AIC', 'BIC', 'NLL', 'loss'] \
                    else fit_result[score_name]             # AIC/BIC/NLL/loss are smaller the better

                # Update best-performing model and its associated statistics
                if score > best_val_score:
                    best_val_score, best_model, best_performance = score, M, dict(sorted(fit_result.items()))
                    best_train_TFPR, best_val_TFPR = train_tfpr_pair, val_tfpr_pair
                    # Prepare a partial function for subsequent calculation of SHAP values
                    self.partial_SHAP[param] = partial(GradientShap(M).attribute,
                                                       baselines=torch.tensor(X_train, dtype=torch.float32))

            # Store the best-performing model and its associated statistics for each param in A2
            self.models_dict[param] = best_model
            self.models_performance_dict[param] = best_performance
            self.TFPR_dict['Train'][param] = best_train_TFPR
            self.TFPR_dict['Val'][param] = best_val_TFPR
            self.is_fitted = True

    def evaluate(self, X, y, prefix='Test'):
        # Type check for X, y, and prefix will be performed in dl_base.test_model.
        assert self.is_fitted, \
            'Call .fit before before evaluation.'
        if self.is_evaluated:
            existing_prefixes = set(k.split('_')[0] for k in self.models_performance_df.keys())
            assert prefix not in existing_prefixes, \
                f'The model was evaluated with the same prefix (={prefix}) before. Use a different prefix.'

        # Store TPR and FPR rates
        self.TFPR_dict[prefix] = {}

        # Evaluate model for each number of hidden units in the given grid
        for param in self.param_grid:
            M = self.models_dict[param]
            n_params = M.get_n_params()
            test_dict, y_pred = test_model(M, X, y, self.criterion, prefix=f'{prefix}_', return_pred=True)
            y_pred = y_pred.cpu().numpy()
            y_pred_lab = np.where(y_pred < .5, 0, 1) if self.n_classes == 2 else np.argmax(y_pred, axis=1)
            test_dict |= classify_metrics(y, y_pred_lab, prefix=f'{prefix}_')
            auroc_, tpr_, fpr_ = classify_AUROC(y, y_pred, prefix=f'{prefix}_', auroc_only=False)
            test_dict |= auroc_
            self.TFPR_dict[prefix][param] = [tpr_, fpr_]
            test_dict |= classify_AIC_BIC(y, y_pred, n_params=n_params, prefix=f'{prefix}_')
            self.models_performance_dict[param] |= dict(sorted(test_dict.items()))
        self.is_evaluated = True

    def get_performance(self):
        assert self.is_fitted, \
            'Call .fit before obtaining performance statistics.'
        if not self.is_evaluated:
            warnings.warn('You may want to call .evaluate before obtaining performance statistics.')

        # Create a pandas.DataFrame to store all the performance statistics
        colnames = ['param'] + list(list(self.models_performance_dict.values())[0].keys())
        df = pd.DataFrame([], columns=colnames)
        for param in sorted(self.models_performance_dict.keys()):
            df.loc[len(df)] = [param] + list(self.models_performance_dict[param].values())
        df['param'] = df['param'].astype(int)
        df['Elapsed_train_epochs'] = df['Elapsed_train_epochs'].astype(int)
        self.models_performance_df = df
        return df

    def get_best_performance(self,
                             partition: str,
                             metric: Literal['AIC', 'AUROC', 'Accuracy', 'BIC', 'F1',
                                             'NLL', 'Precision', 'Recall', 'Specificity', 'loss']):
        # Type and value check
        assert partition in self.TFPR_dict.keys(), \
            f"partition (={partition}) has not been created."
        assert metric in self.metrics_list, \
            f"metric (={metric}) is not supported. Call .metric_list for a list of supported metrics."

        # Obtain the associated characteristics/performance statistics of the best performing model
        df = self.get_performance() if self.models_performance_df is None else self.models_performance_df
        best_model_stat = df.loc[df[f'{partition}_{metric}'].argmin(), :].to_dict() \
            if metric in ['AIC', 'BIC', 'NLL', 'loss'] \
            else df.loc[df[f'{partition}_{metric}'].argmax(), :].to_dict()   # AIC/BIC/NLL/loss are smaller the better
        return best_model_stat

    def get_model(self,
                  param: int):
        # Type and value check
        assert param in self.models_dict.keys(), \
            f"param (={param}) was not found in param_grid (specified when you created a RiskPath model.)"
        assert self.is_fitted, \
            'Call .fit before obtaining a fitted model.'
        return self.models_dict[param]

    def get_best_model(self,
                       partition: str,
                       metric: Literal['AIC', 'AUROC', 'Accuracy', 'BIC', 'F1',
                                       'NLL', 'Precision', 'Recall', 'Specificity', 'loss']):
        # Type and value check performed in .get_best_performance
        best_param = self.get_best_performance(partition, metric)['param']
        return self.models_dict[best_param]

    def get_TPR_FPR(self,
                    param: int,
                    partition: str):
        # Type and value check
        assert param in self.models_dict.keys(), \
            f"param (={param}) was not found in param_grid (specified when you created a RiskPath model.)"
        assert partition in self.TFPR_dict.keys(), f"partition (={partition}) has not been used before."
        return self.TFPR_dict[partition][param]

    def get_SHAP(self,
                 param: int,
                 X: torch.Tensor):
        # Type and value check
        assert self.is_fitted, \
            'Call .fit before obtaining a fitted model.'
        assert param in self.partial_SHAP.keys(), \
            f"param (={param}) was not found in .partial_SHAP.)"
        try:
            X = torch.Tensor(X)
        except:
            raise TypeError(f'X_train must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert X.shape[-1] == self.n_features, \
            (f"The number of features in X (={X.shape[-1]}) must match that used to "
             f"train the model (={self.n_features}).")
        if len(X.shape) == 3:
            assert X.shape[1] == self.n_timestamps, \
                (f"The number of timestamps in X (={X.shape[1]}) must match that used to "
                 f"train the model (={self.n_timestamps}).")

        torch.backends.cudnn.enabled = False            # Disable CUDNN before computing SHAP values
        if self.n_classes == 2:      # For binary classification cases, only one matrix will be returned.
            attributes = self.partial_SHAP[param](inputs=X).cpu().detach().numpy()
        else:                        # For multi-class classification cases, the matrix of each class will be returned.
            attributes = [self.partial_SHAP[param](inputs=X, target=class_idx).cpu().detach().numpy()
                          for class_idx in range(self.n_classes)]
        torch.backends.cudnn.enabled = True             # Re-enable CUDNN
        return attributes

########################################################################################################################
# Define a RiskPath model class for classification
########################################################################################################################


class RPRegressor:
    """
    A PyTorch RiskPath class for regression.

    A. Runtime parameters
    ---------------------
    A1. base_model: An object of any class in the list [ANN_Regressor, LSTM_Regressor, Transformer_Regressor,
                    TCN_Regressor].
        The embedded deep-learning model to be used.
    A2. param_grid: A list.
        The list of values of the tunable parameter of the base_model in A1. Below are the corresponding parameters:
        ANN_Regressor: n_units
        LSTM_Regressor: n_units
        Transformer_Regressor: d_model
        TCN_Regressor: n_units
    A3. verbose: An integer in [0, 1].
        Verbosity. No logging if 0, and information about the grid search over A2 will be displayed if 1.
        Default setting: verbose=1
    A4. kwargs: (Any extra runtime parameters of base_model)
        Example: bidirectional=True when base_model is an LSTM_Classifier object
        Example: n_heads=16 when base_model is a Transformer_Classifier object

    B. Attributes
    -------------
    B1. metrics_list: The list of available evaluation metrics to be used.
    B2. is_fitted: A boolean indicating whether the RiskPath model has been fitted.
    B3. is_evaluated: A boolean indicating whether the RiskPath model has been evaluated.
    B4. n_features: A positive integer indicating the number of features when fitting the RiskPath model.
    B5. n_timestamps: A positive integer indicating the number timestamps when fitting the RiskPath model.
    B6. criterion: A loss function from torch.nn when fitting the RiskPath model.
    B7. models_dict: A dictionary with keys as numbers of units in A2 and values as the corresponding fitted
                     ANN/LSTM model.
    B8. models_performance_dict: A dictionary with keys as numbers of units in A2 and values and dictionary of
                                 performance statistics.
    B9. models_performance_df: A pandas.DataFrame with rows corresponding to the numbers of units in A2 and columns as
                               the performance statistics.
    B10. partial_SHAP: A dictionary with keys as the numbers of units in A2 and values as the partial functions used to
                       compute SHAP values when applying get_SHAP in CX.
    (A1-A4 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. fit(X, y, criterion, optimizer, cv, n_epochs, earlyStopper, verbose_epoch, val_metric, random_state, kwargs)
        Fit each model created according to the grid in A2 with the feature set X and target y.
        :param X: A 2- or 3-dimensional numpy array (or Torch.Tensor).
               Samples of the feature set with dimension as (sample size, number of features) or (sample size,
               number of timestamps, number of features).
        :param y: A 1-dimensional numpy array (or Torch.Tensor).
               Samples of the target with dimension as (sample size,).
        :param criterion: A loss function from torch.nn.
               See https://pytorch.org/docs/main/nn.html#loss-functions. Make sure you specify it WITH brackets.
               Example: torch.nn.NLLLoss()
        :param optimizer: An torch.optim.Optimizer object.
               See https://pytorch.org/docs/stable/optim.html. Make sure you specify it WITHOUT brackets.
               Example: torch.nn.AdamW
        :param cv: An integer.
               The number of folds used in the cross-validation splitting strategy.
               Default setting: cv=5
        :param n_epochs: A positive integer.
               The number of maximum epochs to be run in training the model.
               Default setting: n_epochs=100
        :param earlyStopper: An EarlyStopping object or None.
               It aims to speed up training time and prevent over-fitting.
               Default setting: earlyStopper=EarlyStopping
        :param verbose_epoch: A positive integer or None.
               If integer, it controls the frequency of printing the training and validation losses with a rate of every
               {verbose_epoch} epochs. No logging will be printed if None.
               Default setting: verbose_epoch=None
        :param val_metric: A string in B1.
               Specify the metric used to judge which model in the cross-validation process is the best.
               Default setting: val_metric='MSE'
        :param random_state: An integer, a numpy.random.RandomState instance, or None.
               The seed of the random number generator used in the cross-validation process.
               Default setting: random_state=None
        :param kwargs: (Any extra runtime parameters of optimizer)
               Example: lr=0.001 for the learning rate parameter of the optimizer.
    C2. evaluate(X, y):
        Evaluate the models fitted in C2 with the feature set X and the target y.
        :param X: A 2- or 3-dimensional numpy array (or Torch.Tensor).
               Samples of the feature set with dimension as (sample size, number of features) or (sample size,
               number of timestamps, number of features).
        :param y: A 1-dimensional numpy array (or Torch.Tensor).
               Samples of the target with dimension as (sample size,).
    C3. get_performance()
        Obtain a pandas.DataFrame summarizing the performance statistics of each fitted (and evaluated) model.
        :return: A Pandas DataFrame with rows as models and columns as evaluation metrics.
    C4. get_best_performance(partition, metric)
        Obtain a dictionary from the row of C3 with the best user-specified metric.
        :param partition: A string in ['Train', 'Val'] or any user-specified string used in evaluation.
               Identify which partition of the dataset where the metric is evaluated at.
        :param metric: A string in B1.
               The name of the evaluation metric.
        :return: A dictionary with keys as names of characteristics or metrics of the model and values as the values of
                 those characteristics or metrics.
    C5. get model(param)
        Obtain the fitted model of the specified number of units.
        :param param: A positive integer in the grid in A2.
        :return: The fitted model of the user-specified number of units.
    C6. get_best_model(partition, metric)
        Obtain the best-fitting model according to the specified metric (relative to the given partition).
        :param partition: A string in ['Train', 'Val'] or any user-specified string used in evaluation.
               Identify which partition of the dataset where the metric is evaluated at.
        :param metric: metric: A string in B1.
               The name of the evaluation metric.
        :return: The fitted model of the user-specified metric.
    C7. get_SHAP(param, X)
        Compute the SHAP values (for each class) by explaining X with the model identified by param (where the
        explanation baseline is the dataset used to train the model).
        :param param: A positive integer in the grid in A2.
        :param X: A 2- or 3-dimensional numpy array (or Torch.Tensor).
               Samples of the feature set with dimension as (sample size, number of features) or (sample size,
               number of timestamps, number of features).
        :return: A 2- or 3-dimensional numpy array of SHAP values (depending on the dimension of X).

    D. Remarks
    ----------
    Runtime parameter settings (in A4) of the base_model (in A1) are as follows:
    ANN_Regressor: n_layers (default: 2)
    LSTM_Regressor: n_layers (default: 2), bidirectional (default: False)
    Transformer_Regressor: n_layers (default: 2), n_units (default: 1024), n_heads (default: 8)
    TCN_Regressor: n_layers (default: 2), kernel_size (default: 3)
    """
    def __init__(self,
                 base_model: Union[ANN_Regressor, LSTM_Regressor, Transformer_Regressor, TCN_Regressor],
                 param_grid: list[int],
                 verbose: int = 1,
                 **kwargs):

        # Type and value check
        assert base_model in [ANN_Regressor, LSTM_Regressor, Transformer_Regressor, TCN_Regressor], \
            (f'model must be an object from [ANN_Regressor, LSTM_Regressor, Transformer_Regressor, TCN_Regressor]. '
             f'Now its type is {type(base_model)}.')
        self.base_model = base_model
        try:
            param_grid = list(param_grid)
        except:
            raise TypeError(f'param_grid must be (convertible to) a list. Now its type is {type(param_grid)}.')
        assert all([isinstance(param, int) and param > 0 for param in param_grid]), \
            f'All elements in param_grid must be a positive integer.'
        self.param_grid = param_grid
        assert verbose in [0, 1], \
            f"verbose must be in [0, 1]. Now its value is {verbose}."
        self.verbose = verbose
        self.kwargs = kwargs

        # Define attributes
        self.metrics_list = ['AIC', 'BIC', 'MAE', 'MAPE', 'MSE', 'R2', 'RMSE', 'loss']
        self.is_fitted = False              # Boolean indicating whether the model has been fitted
        self.is_evaluated = False           # Boolean indicating whether the model has been evaluated
        self.n_features = None              # Number of features to be known from the data when fitted.
        self.n_timestamps = None            # Number of timestamps to be known from the data when fitted.
        self.criterion = None               # Criterion (aka loss) for back-propagation to be input when fitted.
        self.models_dict = {}               # Store the fitted models (only best in k-fold)
        self.models_performance_dict = {}   # Store the performance of the fitted models (only best in k-fold)
        self.models_performance_df = None   # Similar to .models_performance_dict but in Pandas.DataFrame
        self.partial_SHAP = {}              # Store the partial functions for SHAP calculation

    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            criterion: LossFunction,
            optimizer: torch.optim.Optimizer,
            cv: int = 5,
            n_epochs: int = 100,
            earlyStopper: Optional[EarlyStopping] = None,
            verbose_epoch: Optional[int] = None,
            val_metric: Literal['AIC', 'BIC', 'MAE', 'MAPE', 'MSE', 'R2', 'RMSE', 'loss'] = 'MSE',
            random_state: Optional[Union[int, np.random.RandomState]] = None,
            **kwargs):
        # Type check for X and y, n_epochs, earlyStopper, verbose_epoch will be performed in dl_base.train_model.
        # No type check for criterion and optimizer because PyTorch did not define the associated class.
        assert isinstance(cv, int), \
            f'cv has to be a positive integer. Now its type is {type(cv)}.'
        assert cv >= 1, \
            f'cv has to be a positive integer. Now it is {cv}.'
        assert val_metric in self.metrics_list, \
            f'val_metric must be in {self.metrics_list}. Now it is {val_metric}.'

        # Create indices for cross-validation (and rely on sklearn to check if random_state is valid)
        kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)

        # Update attributes from training data
        self.n_features = X.shape[-1]
        if len(X.shape) == 3:
            self.n_timestamps = X.shape[1]
        self.criterion = criterion

        # Start grid search
        for param in self.param_grid:

            # Storing characteristics of the best-performing model during cross-validation
            best_val_score, best_model, best_performance = float('inf'), None, None

            # Start cross-validation
            for k_idx, (train, val) in enumerate(kf.split(X, y), 1):
                X_train, X_val = np.take(X, train, axis=0), np.take(X, val, axis=0)
                y_train, y_val = np.take(y, train), np.take(y, val)


                # Create prediction model
                if self.base_model == ANN_Classifier:
                    M = self.base_model(n_feat=X_train.shape[-1], n_units=int(param))
                elif self.base_model == LSTM_Classifier:
                    M = self.base_model(n_feat=X_train.shape[-1], n_units=int(param)
                                        **self.kwargs)
                elif self.base_model == Transformer_Classifier:
                    M = self.base_model(n_feat=X_train.shape[-1], n_timestamps=self.n_timestamps, d_model=int(param),
                                        **self.kwargs)
                elif self.base_model == TCN_Classifier:
                    M = self.base_model(n_feat=X_train.shape[-1], n_units=int(param),
                                        **self.kwargs)
                else:
                    M = None    # Unnecessary but here for consistency

                # Configure prediction model
                M.set_device('cuda:0' if torch.cuda.is_available() else 'cpu')
                M.init_Xavier_weights()

                # Train models by k-fold validation
                if self.verbose == 1:
                    print(f"\nTraining {k_idx}/{cv} model with param={param}...", flush=True)
                fit_result = train_model(M, X_train, y_train, X_val, y_val, n_epochs,
                                         criterion, optimizer, earlyStopper, verbose_epoch, **kwargs)
                if earlyStopper is not None:
                    earlyStopper.reset()
                n_params = M.get_n_params()     # For AIC/BIC calculation

                # Obtain training and validation performance statistics
                for (X_, y_, prefix) in [(X_train, y_train, 'Train_'), (X_val, y_val, 'Val_')]:
                    loss, y_pred_ = test_model(M, X_, y_, criterion, prefix=prefix, return_pred=True)
                    y_pred_ = y_pred_.cpu().numpy()
                    fit_result |= regress_metrics(y_, y_pred_, prefix=prefix)
                    fit_result |= regress_AIC_BIC(y_, y_pred_, n_params=n_params, prefix=prefix)

                # Select the metric to compare model performance
                score_name = f'Val_{val_metric}'
                score = fit_result[score_name] if val_metric != 'R2' \
                    else -fit_result[score_name]                        # R2 is larger the better

                # Update best-performing model and its associated statistics
                if score < best_val_score:
                    best_val_score, best_model, best_performance = score, M, dict(sorted(fit_result.items()))

                    # Prepare a partial function for subsequent calculation of SHAP values
                    self.partial_SHAP[param] = partial(GradientShap(M).attribute,
                                                       baselines=torch.tensor(X_train, dtype=torch.float32))

            # Store the best-performing model and its associated statistics for each value of param in A2
            self.models_dict[param] = best_model
            self.models_performance_dict[param] = best_performance
            self.is_fitted = True

    def evaluate(self, X, y, prefix='Test'):
        # Type check for X, y, and prefix will be performed in dl_base.test_model.
        assert self.is_fitted, \
            'Call .fit before before evaluation.'
        if self.is_evaluated:
            existing_prefixes = set(k.split('_')[0] for k in self.models_performance_df.keys())
            assert prefix not in existing_prefixes, \
                f'The model was evaluated with the same prefix (={prefix}) before. Use a different prefix.'

        # Evaluate model for each number of hidden units in the given grid
        for param in self.param_grid:
            M = self.models_dict[param]
            n_params = M.get_n_params()
            test_dict, y_pred = test_model(M, X, y, self.criterion, prefix=f'{prefix}_', return_pred=True)
            y_pred = y_pred.cpu().numpy()
            test_dict |= regress_metrics(y, y_pred, prefix=f'{prefix}_')
            test_dict |= regress_AIC_BIC(y, y_pred, n_params=n_params, prefix=f'{prefix}_')
            self.models_performance_dict[param] |= dict(sorted(test_dict.items()))
        self.is_evaluated = True

    def get_performance(self):
        assert self.is_fitted, \
            'Call .fit before obtaining performance statistics.'
        if not self.is_evaluated:
            warnings.warn('You may want to call .evaluate before obtaining performance statistics.')

        # Create a pandas.DataFrame to store all the performance statistics
        colnames = ['param'] + list(list(self.models_performance_dict.values())[0].keys())
        df = pd.DataFrame([], columns=colnames)
        for param in sorted(self.models_performance_dict.keys()):
            df.loc[len(df)] = [param] + list(self.models_performance_dict[param].values())
        df['param'] = df['param'].astype(int)
        df['Elapsed_train_epochs'] = df['Elapsed_train_epochs'].astype(int)
        self.models_performance_df = df
        return df

    def get_best_performance(self,
                             partition: str,
                             metric: Literal['AIC', 'BIC', 'MAE', 'MAPE', 'MSE', 'R2', 'RMSE', 'loss']):
        # Type and value check
        existing_prefixes = set(k.split('_')[0] for k in self.models_performance_df.keys())
        assert partition in existing_prefixes, \
            f"partition (={partition}) has not been created."
        assert metric in self.metrics_list, \
            f"metric (={metric}) is not supported. Call .metric_list for a list of supported metrics."

        # Obtain the associated characteristics/performance statistics of the best performing model
        df = self.get_performance() if self.models_performance_df is None else self.models_performance_df
        best_model_stat = df.loc[df[f'{partition}_{metric}'].argmin(), :].to_dict() if metric != 'R2' \
            else df.loc[df[f'{partition}_{metric}'].argmax(), :].to_dict()   # R2 is greater the better
        return best_model_stat

    def get_model(self,
                  param: int):
        # Type and value check
        assert param in self.models_dict.keys(), \
            f"param (={param}) was not found in param_grid (specified when you created a RiskPath model.)"
        assert self.is_fitted, \
            'Call .fit before obtaining a fitted model.'
        return self.models_dict[param]

    def get_best_model(self,
                       partition: str,
                       metric: Literal['AIC', 'BIC', 'MAE', 'MAPE', 'MSE', 'R2', 'RMSE', 'loss']):
        # Type and value check performed in .get_best_performance
        best_param = self.get_best_performance(partition, metric)['param']
        return self.models_dict[best_param]

    def get_SHAP(self,
                 param: int,
                 X: torch.Tensor):
        # Type and value check
        assert self.is_fitted, \
            'Call .fit before obtaining a fitted model.'
        assert param in self.partial_SHAP.keys(), \
            f"param (={param}) was not found in .partial_SHAP.)"
        try:
            X = torch.Tensor(X)
        except:
            raise TypeError(f'X_train must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert X.shape[-1] == self.n_features, \
            (f"The number of features in X (={X.shape[-1]}) must match that used to "
             f"train the model (={self.n_features}).")
        if len(X.shape) == 3:
            assert X.shape[1] == self.n_timestamps, \
                (f"The number of timestamps in X (={X.shape[1]}) must match that used to "
                 f"train the model (={self.n_timestamps}).")

        torch.backends.cudnn.enabled = False        # Disable CUDNN before computing SHAP values
        attributes = self.partial_SHAP[param](inputs=X).cpu().detach().numpy()
        torch.backends.cudnn.enabled = True         # Re-enable CUDNN
        return attributes

########################################################################################################################
