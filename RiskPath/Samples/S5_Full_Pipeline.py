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
from DL.algo import Transformer_Classifier
from DL.riskpath import RPClassifier
from DL.plots import plot_AUROC, plot_SHAP_3D_movie
from Preprocess.concatenate import TSConcatenate
from Preprocess.feature_select import FSClassifier
from Utils.log import create_log
from Utils.simulators import sample_dataset_2

########################################################################################################################
# Experiment on RiskPath with the optional procedures of feature selection and timestamp concatenation
########################################################################################################################

# Create logging file
filename = __file__.split('.py')[0].split('/')[-1]
create_log(filename)

# Simulate a toy dataset for binary classification
X_train_list, X_test_list, y_train, y_test, feature_names = sample_dataset_2()

# Some features (e.g., X3) have no samples in some (but not all) timestamps
feat_example = sorted(set(X_train_list[0].columns).difference(X_train_list[1].columns))[0]
print(f"Is {feat_example} a feature in the 0-th timestamp: {feat_example in X_train_list[0].columns}")
print(f"Is {feat_example} a feature in the 1-th timestamp: {feat_example in X_train_list[1].columns}", )

# Perform feature selection for the training feature dataset in each timestamp
relevant_idx_lists = []
for t in range(len(X_train_list)):
    print(f"Feature selection of the training dataset in the {t}-th timestamp.")
    FSC = FSClassifier(B_max_iter=50, B_verbose=0)
    FSC.fit_both(X_train_list[t], y_train)
    relevant_idx_lists.append(FSC.get_rlv_feat())          # Record the relevant feature indices

# Perform data imputation and timestamp concatenation
TSC_train = TSConcatenate(verbose=0)
TSC_train.load_data_in_order(X_train_list); TSC_train.load_fs_result_in_order(relevant_idx_lists)
TSC_train.concatenate()
X_train_3D, feature_names, imputed = TSC_train.get_result()

# Do the same for the test partition
TSC_test = TSConcatenate(verbose=0)
TSC_test.load_data_in_order(X_test_list); TSC_test.load_fs_result_in_order(relevant_idx_lists)
TSC_test.concatenate()
X_test_3D, _, _ = TSC_test.get_result()

# Now we can proceed to RiskPath model prediction
RPC = RPClassifier(base_model=Transformer_Classifier, param_grid=[16, 32, 64])

RPC.fit(X_train_3D, y_train, cv=3, n_epochs=50,
        criterion=torch.nn.BCELoss(),
        optimizer=torch.optim.SGD,
        val_metric='AUROC')

RPC.evaluate(X_test_3D, y_test)
best_model_stat = RPC.get_best_performance(partition='Test', metric='AUROC')
best_param = best_model_stat['param']
plot_AUROC(*RPC.get_TPR_FPR(best_param, 'Test'))

# Compute the SHAP values evaluated by the best-AUROC model at the test partition
shap = RPC.get_SHAP(param=best_param, X=X_test_3D)

# Create a 3D animation illustrating how SHAP values vary across timestamps
plot_SHAP_3D_movie(shap, feature_names=feature_names, top_n_features=10)
