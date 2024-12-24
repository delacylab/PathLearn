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
import torch
from DL.algo import TCN_Classifier
from DL.plots import plot_AUROC, plot_SHAP_3D_movie
from DL.riskpath import RPClassifier
from Utils.timeseries_simulators import sample_dataset_1

########################################################################################################################
# Experiment on RiskPath with the optional procedures of feature selection and timestamp concatenation
########################################################################################################################

# Create a partitioned sample dataset
X_train, X_test, y_train, y_test, feature_names = sample_dataset_1()

# Create, fit, and evaluate a RiskPath model (embedded with Transformer)
# RPC = RPClassifier(base_model=LSTM_Classifier, param_grid=[16, 32, 64])
# RPC = RPClassifier(base_model=Transformer_Classifier, param_grid=[16, 32, 64])
RPC = RPClassifier(base_model=TCN_Classifier, param_grid=[16, 32, 64])
RPC.fit(X_train, y_train, criterion=torch.nn.BCELoss(), optimizer=torch.optim.SGD)
RPC.evaluate(X_test, y_test)

# Identify the parameter associated with the best AUROC in the test partition
best_param = RPC.get_best_performance(partition='Test', metric='AUROC')['param']
plot_AUROC(*RPC.get_TPR_FPR(best_param, 'Test'))

# Compute the SHAP values evaluated by the best-AUROC model at the test partition
shap = RPC.get_SHAP(param=best_param, X=X_test)

# Create a 3D animation illustrating how SHAP values vary across timestamps
plot_SHAP_3D_movie(shap, feature_names=feature_names, top_n_features=10, filename='S4_RiskPath_Pipeline')

########################################################################################################################
