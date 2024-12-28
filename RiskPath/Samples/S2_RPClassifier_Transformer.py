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
import torch.nn
from ..DL.algo import Transformer_Classifier, EarlyStopping
from ..DL.riskpath import RP_Classifier
from ..Utils.logger import create_log
from ..Utils.plots import (plot_performance, plot_AUROC, plot_mean_predictor_importance, plot_predictor_path,
                         plot_epoch_importance, plot_shap_bar, plot_shap_beeswarm, plot_shap_heatmap, plot_shap_movie)
from ..Utils.timeseries_simulators import make_ts_classification
from sklearn.model_selection import train_test_split

########################################################################################################################
# Experiment on RiskPath for binary classification
########################################################################################################################

# Create logging file
create_log(__file__.split('.py')[0])

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

# Create a RiskPath model (with a Transformer base model)
RPC = RP_Classifier(base_model=Transformer_Classifier,
                    param_grid=[120, 240, 360, 480])

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

# Obtain and plot overall performance statistics by accuracy
df = RPC.get_performance()
plot_performance(df=df, rename_dict={'param': 'Width of models', 'Train': 'Training'})

# Identify the best (test) AUROC model
best_model_stat = RPC.get_best_performance(partition='Test', metric='AUROC')
best_param = best_model_stat['param']

# Obtain and plot AUROC
TPR, FPR = RPC.get_TPR_FPR(best_param, 'Test')
plot_AUROC(TPR, FPR)

# Obtain and plot mean predictor importance (= mean-absolute SHAP across samples & time epochs)
mpi = RPC.get_SHAP(param=best_param, X=X_test, average='Sample_Epoch')
plot_mean_predictor_importance(mpi)

# Obtain and plot predictor path (= mean-absolute SHAP across samples)
pp = RPC.get_SHAP(param=best_param, X=X_test, average='Sample')
plot_predictor_path(pp, feature_names, top_n_features=20, y_log=False)

# Obtain and plot epoch importance (=mean-absolute SHAP across samples and features)
ei = RPC.get_SHAP(param=best_param, X=X_test, average='Sample_Feature')
plot_epoch_importance(ei)

# Obtain and plot mean predictor importance in the last epoch
mpi = RPC.get_SHAP(param=best_param, X=X_test, average='Sample')
mpi = mpi[-1, :]
plot_shap_bar(mpi, feature_names=feature_names)

# Obtain the SHAP values in the last epoch and show its beeswarm/strip plot
shap = RPC.get_SHAP(param=best_param, X=X_test)
shap_last, X_test_last = shap[:, -1, :], X_test[:, -1, :]
plot_shap_beeswarm(shap_last, X_test_last, feature_names)

# Obtain the SHAP values of a specific feature and show its heatmap
plot_shap_heatmap(shap, 0, True)

# Show 3D animation
plot_shap_movie(shap, feature_names=feature_names, top_n_features=20)

#######################################################################################################################
