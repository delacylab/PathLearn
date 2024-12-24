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
import torch.nn
from DL.algo import TCN_Classifier, EarlyStopping
from DL.plots import (plot_performance, plot_AUROC, plot_MASHAP_trend, plot_SHAP_strip, plot_MASHAP_feat_trend,
                      plot_SHAP_feat_heatmap, plot_MASHAP_bar, plot_SHAP_3D_movie)
from DL.riskpath import RPClassifier
from sklearn.model_selection import train_test_split
from Utils.logger import create_log
from Utils.timeseries_simulators import make_ts_classification

########################################################################################################################
# Experiment on RiskPath for binary classification
########################################################################################################################

# Create logging file
filename = __file__.split('.py')[0].split('/')[-1]
create_log(filename)

# Simulate the dataset
X, y = make_ts_classification(n_samples_per_class=500,
                              n_timestamps=10,
                              n_features=30,
                              n_informative=10,
                              n_classes=2,
                              noise_level=10,
                              random_state=42)
feature_names = [f'X_{i + 1}' for i in range(X.shape[2])]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create a RiskPath model (with an TCN base model)
RPC = RPClassifier(base_model=TCN_Classifier,
                   param_grid=[100, 200, 300],
                   n_layers=2,
                   kernel_size=3)

# Fit the RiskPath model
RPC.fit(X_train, y_train,
        cv=3,
        n_epochs=50,
        criterion=torch.nn.BCELoss(),
        optimizer=torch.optim.SGD,
        earlyStopper=EarlyStopping(patience=5),
        verbose_epoch=10,
        val_metric='AUROC',
        lr=0.005)

# Evaluate the RiskPath model
RPC.evaluate(X_test, y_test)

# Get overall performance statistics and plot
df = RPC.get_performance()
plot_performance(df=df,
                 var_column='param',
                 prefixes=['Train', 'Val', 'Test'],
                 metric='AUROC',
                 sep='_',
                 title=None,
                 filename=None,
                 rename_dict={'param': 'Dimension of embedding vector', 'Train': 'Training', 'Val': 'Validation'})

# Identify the best (test) AUROC model
best_model_stat = RPC.get_best_performance(partition='Test', metric='AUROC')
best_n_units = best_model_stat['param']

# Plot AUROC curve
TPR, FPR = RPC.get_TPR_FPR(best_n_units, 'Test')
plot_AUROC(TPR, FPR)

# Get SHAP values and mean-absolute SHAP across timestamps
attr = RPC.get_SHAP(param=best_n_units, X=X_test)
attr_tsMA = np.mean(np.abs(attr), axis=1)

# Show the SHAP trend plot (with SHAP values averaged over timestamps first)
plot_MASHAP_trend(attr_tsMA)

# Show SHAP strip plot in the last timestamp
plot_SHAP_strip(attr[:, -1, :], X_test[:, -1, :], feature_names,
                top_n_features=20, max_points=200, random_state=42)

# Show features' MA-SHAP trend curve
plot_MASHAP_feat_trend(attr, feature_names, top_n_features=20, y_log=False)

# Show a feature's SHAP heatmap
plot_SHAP_feat_heatmap(attr, feature_idx=0, scale_intensity=True)

# Show a stacked SHAP bar chat
plot_MASHAP_bar(attr_tsMA, feature_names, top_n_features=20, stack=True)

# Show 3D animation
plot_SHAP_3D_movie(attr, feature_names=feature_names, top_n_features=20, filename=f"{filename}_3D")

########################################################################################################################
