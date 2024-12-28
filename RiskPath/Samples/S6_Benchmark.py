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
from ..Utils.benchmarking import Benchmark_Classifier
from ..Utils.plots import plot_AUROC
from ..Utils.timeseries_simulators import make_ts_classification
from ..Utils.wavelet_transform import extract_wavelet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

########################################################################################################################
# Experiment on RiskPath for binary classification
########################################################################################################################

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
X_train, X_test = extract_wavelet(X_train), extract_wavelet(X_test)

# Create a Benchmarking model (with an SVM base model)
M = Benchmark_Classifier(base_model=SVC(probability=True, random_state=42))
M.fit(X_train, y_train)
M.evaluate(X_test, y_test)
plot_AUROC(*M.TFPR_dict['Test'])
