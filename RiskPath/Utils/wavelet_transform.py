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
import pywt

########################################################################################################################


def extract_wavelet(data, wavelet='haar', level=1):
    """
    Extract wavelet features from time-series data.
    :param data: A numpy ndarray.
           A 3-dimensional dataset with dimensions (number of samples, number of timestamps, number of features).
    :param wavelet: A string.
           The wavelet function to use.
           Default setting: wavelet='haar'
    :param level: An integer.
           A positive integer referring to the decomposition level. The maximum number of level is determined by the
           formula log_2 (number of timestamps). For short time-series (e.g., 4-10 timestamps), use level in [1, 2].
           Default setting: level=1
    :return: A numpy ndarray.
             The 2-dimensional dataset with dimensions (number of samples, features * wavelet_coefficients) where
             wavelet_coefficients is determined by the wavelet function, decomposition level, and the length of the
             original time-series.
    """
    all_features = []
    for sample in data:
        sample_features = []
        for feature in sample.T:
            coeffs = pywt.wavedec(feature, wavelet=wavelet, level=level)
            sample_features.extend(np.hstack(coeffs))
        all_features.append(sample_features)
    return np.array(all_features)

########################################################################################################################
