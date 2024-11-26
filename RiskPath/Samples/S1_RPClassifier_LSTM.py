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
from DL.algo import LSTM_Classifier, EarlyStopping
from DL.plots import (plot_performance, plot_AUROC, plot_MASHAP_trend, plot_SHAP_strip, plot_MASHAP_feat_trend,
                      plot_SHAP_feat_heatmap, plot_MASHAP_bar, plot_SHAP_3D_movie)
from DL.riskpath import RPClassifier
from sklearn.model_selection import train_test_split
from Utils.log import create_log
from Utils.simulators import make_ts_classification

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

# Create a RiskPath model (with an LSTM base model)
RPC = RPClassifier(base_model=LSTM_Classifier,
                   param_grid=(30, 40),
                   bidirectional=False)

# Fit the RiskPath model
RPC.fit(X_train, y_train,
        cv=3,
        n_epochs=50,
        criterion=torch.nn.BCELoss(),
        optimizer=torch.optim.SGD,
        earlyStopper=EarlyStopping(patience=5),
        val_metric='AUROC',
        lr=0.005)

# Evaluate the RiskPath model
RPC.evaluate(X_test, y_test)

# Get overall performance statistics and plot
df = RPC.get_performance()
plot_performance(df=df,
                 var_column='param',
                 prefixes=['Train', 'Val', 'Test'],
                 metric='Accuracy',
                 sep='_',
                 rename_dict={'param': 'Number of hidden units', 'Train': 'Training', 'Val': 'Validation'},
                 filename=filename+'_01_Perform')

# Identify the best (test) AUROC model
best_model_stat = RPC.get_best_performance(partition='Test', metric='AUROC')
best_n_units = best_model_stat['param']

# Plot AUROC curve
TPR, FPR = RPC.get_TPR_FPR(best_n_units, 'Test')
plot_AUROC(TPR, FPR, filename=filename+'_02_AUROC')

# Get SHAP values and mean-absolute SHAP across timestamps
attr = RPC.get_SHAP(param=best_n_units, X=X_test)
attr_tsMA = np.mean(np.abs(attr), axis=1)

# Show the SHAP trend plot (with SHAP values averaged over timestamps first)
plot_MASHAP_trend(attr_tsMA, filename=filename+'_03_MASHAP_trend')

# Show features' MA-SHAP trend curve
plot_MASHAP_feat_trend(attr, feature_names, top_n_features=20, y_log=False, filename=filename+'_04_MASHAP_trend_feat')

# Show a stacked SHAP bar chat
plot_MASHAP_bar(attr_tsMA, feature_names, top_n_features=20, stack=True, filename=filename+'_05_MASHAP_bar')

# Show SHAP strip plot in the last timestamp
plot_SHAP_strip(attr[:, -1, :], X_test[:, -1, :], feature_names,
                top_n_features=20, max_points=200, random_state=42, filename=filename+'_06_SHAP_strip')

# Show a feature's SHAP heatmap
plot_SHAP_feat_heatmap(attr, feature_idx=0, scale_intensity=True, filename=filename+'_07_SHAP_heatmap')

# Show 3D animation
plot_SHAP_3D_movie(attr, feature_names=feature_names, top_n_features=20, filename=f"{filename}_3D")

########################################################################################################################
