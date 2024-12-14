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
from time import time
from torch import nn
from typing import Callable, Optional, Union

########################################################################################################################
# Define a callable class for loss functions in PyTorch
########################################################################################################################
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

########################################################################################################################
# Define the early stopping class
########################################################################################################################

class EarlyStopping:
    """
    Early stopping technique used during model training to speed up runtime and prevent over-fitting.

    A. Runtime parameters
    ---------------------
    A1. patience: A non-negative integer.
        Number of times that a worse result (depending on A2) can be tolerated.
        Default setting: patience=1
    A2. min_delta: A non-negative float.
        The threshold where loss_old + min_delta < loss_new is considered as worse.
        Default setting: min_delta=1e-8

    B. Attributes
    -------------
    B1. min_loss: A float, recording the minimum loss encountered in the training process.
    B2. counter: A non-negative integer, recording the number of times a worse result is observed. The counter restarts
                 when a better result (i.e., smaller loss) is observed.
    (A1-A2 are initialized as instance attributes.)

    C. Methods
    ---------
    C1. refresh()
        Reset the attributes B1 and B2 to their original values.
    C2. early_stop(loss)
    :param loss: An integer or float. The loss obtained in a given training epoch.
    :return: A boolean indicating whether the training process should be stopped.
    """
    def __init__(self,
                 patience: int = 1,
                 min_delta: float = 1e-8):

        # Type and value check
        assert isinstance(patience, int), \
            f"patience must be an integer. Now its type is {type(patience)}."
        assert patience >= 0, \
            f"patience must be a non-negative integer. Now it is {patience}."
        self.patience = patience
        try:
            min_delta = float(min_delta)
        except:
            raise TypeError(f"min_delta must be a float. Now its type is {type(min_delta)}.")
        assert min_delta >= 0, \
            f"min_delta must be a non-negative float. Now it is {min_delta}."
        self.min_delta = min_delta
        self.min_loss = float('inf')
        self.counter = 0

    def reset(self):
        self.min_loss = float('inf')
        self.counter = 0

    def early_stop(self, loss: float):
        try:
            loss = float(loss)
        except:
            raise TypeError(f"loss must be a float. Now its type is {type(loss)}.")
        if loss < self.min_loss:
            self.min_loss, self.counter = loss, 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

########################################################################################################################
# Define a superclass (inherited from torch.nn.Module) for various model classes (e.g., ANN, LSTM, Transformer, TCN)
########################################################################################################################

class DL_Class(nn.Module):
    """
    A superclass for various model classes that will be used in this module.

    A. Runtime parameters
    ---------------------
    (None)

    B. Attributes
    -------------
    B1. dummy_param: A torch.nn.Parameter object.
        An identifier of the physical location of the model.

    C. Methods
    ----------
    C1. set_device(device_str)
        :param device_str: A string or torch.device object. The physical location of the model to be set.
    C2. get_device()
        :return: A string. The current physical location of the model.
    C3. get_n_params()
        :return: An integer. The number of parameters of the model.
    """
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def set_device(self, device_str: Union[str, torch.device]):
        assert type(device_str) in [str, torch.device], \
            f'device_str must be a torch.device object or a string. Now its type is {type(device_str)}.'
        self.to(device_str)

    def get_device(self):
        return self.dummy_param.device

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())        # For AIC & BIC calculation

########################################################################################################################
# Define an ANN model class for classification
########################################################################################################################

class ANN_Classifier(DL_Class):
    """
    A PyTorch ANN class for classification.

    A. Runtime parameters
    ---------------------
    A1. n_feat: A positive integer.
        The number of features to be fitted to the model.
    A2. n_units: A positive integer.
        The number of hidden units in each hidden layer.
    A3. n_classes: A positive integer greater than 1.
        The number of classes in the target.
    A4. n_layers: A positive integer.
        The number of hidden layers in the model.
        Default setting: n_layers=2

    B. Attributes
    -------------
    B1. model: A torch.nn.modules.container.Sequential object.
        The actual model.
    (A1-A4 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. forward(X)
        :param X: A 2-dimensional numpy array (or Torch.Tensor) with rows as samples and columns as features.
        :return: A torch.Tensor of dimension (samples size, ) [for binary case] or (sample size, n_classes) [for
                 multiclass case] as the probability measure returned by the model.
    C2. init_Xavier_weights()
        Assign weights with Xavier uniform distribution and bias as 0.
        See https://paperswithcode.com/method/xavier-initialization
    """
    def __init__(self,
                 n_feat: int,
                 n_units: int,
                 n_classes: int,
                 n_layers: int = 2):
        super().__init__()

        # Type and value check
        assert isinstance(n_feat, int), \
            f"n_feat must be a positive integer. Now its type is {type(n_feat)}."
        assert n_feat >= 1, \
            f"n_feat must be a positive integer. Now its value is {n_feat}."
        self.n_feat = n_feat
        assert isinstance(n_units, int), \
            f"n_units must be a positive integer. Now its type is {type(n_units)}."
        assert n_units >= 1, \
            f"n_units must be a positive integer. Now its value is {n_units}."
        self.n_units = n_units
        assert isinstance(n_classes, int), \
            f"n_classes must be a positive integer. Now its type is {type(n_classes)}."
        assert n_classes >= 2, \
            f"n_classes must be a positive integer not less than 2. Now its value is {n_classes}."
        self.n_classes = n_classes
        assert isinstance(n_layers, int), \
            f"n_layers must be a positive integer. Now its type is {type(n_layers)}."
        assert n_layers >= 1, \
            f"n_layers must be a positive integer. Now its value is {n_layers}."
        self.n_layers = n_layers

        # Create model
        modules = [nn.Linear(n_feat, n_units), nn.ReLU()]
        for _ in range(n_layers-1):
            modules += [nn.Linear(n_units, n_units), nn.ReLU()]
        if n_classes == 2:
            modules += [nn.Linear(n_units, 1)]
        else:
            modules += [nn.Linear(n_units, self.n_classes)]
        self.model = nn.Sequential(*modules)

    def forward(self, X: Union[np.ndarray, torch.Tensor]):
        try:
            X = torch.Tensor(X)
        except:
            raise TypeError(f'X must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert len(X.shape) == 2, \
            f'X must be 2-dimensional. Now it is {len(X.shape)}-dimensional.'
        device = self.get_device()
        X = X.to(device)
        output = self.model(X)
        if self.n_classes == 2:
            return torch.reshape(torch.sigmoid(output), shape=(-1,))        # sigmoid for binary classification
        else:
            return torch.nn.functional.softmax(output, dim=1)               # softmax for multiclass classification

    def init_Xavier_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

########################################################################################################################
# Define an ANN model class for regression tasks
########################################################################################################################

class ANN_Regressor(DL_Class):
    """
    A PyTorch ANN class for regression.

    A. Runtime parameters
    ---------------------
    A1. n_feat: A positive integer.
        The number of features to be fitted to the model.
    A2. n_units: A positive integer.
        The number of hidden units in each hidden layer.
    A3. n_layers: A positive integer.
        The number of hidden layers in the model.
        Default setting: n_layers=2
    B. Attributes
    -------------
    B1. model: A torch.nn.modules.container.Sequential object.
        The actual model.
    (A1-A3 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. forward(X)
        :param X: A 2-dimensional numpy array (or Torch.Tensor) with rows as samples and columns as features.
        :return: A torch.Tensor of dimension (samples size, ) as the model output.
    C2. init_Xavier_weights()
        Assign weights with a Xavier uniform distribution and bias as 0.
        See https://paperswithcode.com/method/xavier-initialization
    """
    def __init__(self,
                 n_feat: int,
                 n_units: int,
                 n_layers: int):
        super().__init__()

        # Type and value check
        assert isinstance(n_feat, int), \
            f"n_feat must be a positive integer. Now its type is {type(n_feat)}."
        assert n_feat >= 1, \
            f"n_feat must be a positive integer. Now its value is {n_feat}."
        self.n_feat = n_feat
        assert isinstance(n_units, int), \
            f"n_units must be a positive integer. Now its type is {type(n_units)}."
        assert n_units >= 1, \
            f"n_units must be a positive integer. Now its value is {n_units}."
        self.n_units = n_units
        assert isinstance(n_layers, int), \
            f"n_layers must be a positive integer. Now its type is {type(n_layers)}."
        assert n_layers >= 1, \
            f"n_layers must be a positive integer. Now its value is {n_layers}."
        self.n_layers = n_layers

        # Create model
        modules = [nn.Linear(n_feat, n_units), nn.ReLU()]
        for _ in range(n_layers-1):
            modules += [nn.Linear(n_units, n_units), nn.ReLU()]
        modules += [nn.Linear(n_units, 1)]
        self.model = nn.Sequential(*modules)

    def forward(self, X: Union[np.ndarray, torch.Tensor]):
        try:
            X = torch.Tensor(X)
        except:
            raise TypeError(f'X must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert len(X.shape) == 2, \
            f'X must be 2-dimensional. Now it is {len(X.shape)}-dimensional.'
        device = self.get_device()
        X = X.to(device)
        output = self.model(X).squeeze()
        return output

    def init_Xavier_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

########################################################################################################################
# Define an LSTM model class for classification
# References
# https://www.kaggle.com/code/khalildmk/simple-two-layer-bidirectional-lstm-with-pytorch
# https://medium.com/@reddyyashu20/bidirectional-rnn-python-code-in-keras-and-pytorch-22b9a9a3c034
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_bidirectional_lstm.py
########################################################################################################################

class LSTM_Classifier(DL_Class):
    """
     A PyTorch LSTM class for classification.

     A. Runtime parameters
     ---------------------
     A1. n_feat: A positive integer.
         The number of features to be fitted to the model.
     A2. n_units: A positive integer.
         The number of hidden units in each hidden layer
     A3. n_classes: A positive integer greater than 1.
         The number of classes in the target.
     A4. n_layers: A positive integer.
         The number of hidden layers in the model.
         Default setting: n_layers=2
     A5. bidirectional: A boolean.
         Whether the model will be constructed with a bidirectional layer for each hidden layer.
         Default setting: bidirectional=False

     B. Attributes
     -------------
     B1. model: A torch.nn.modules.container.Sequential object.
         Together with the output_layer in B2, it functions as the actual model .
     B2. output_layer: A torch.nn.Linear object.
         The output layer of the actual model.
     (A1-A5 are initialized as instance attributes.)

     C. Methods
     ----------
     C1. forward(X)
         :param X: A 3-dimensional numpy array (or Torch.Tensor) with dimension as (sample size, number of timestamps,
                   number of features)
         :return: A torch.Tensor of dimension (samples size, ) [for binary case] or (sample size, n_classes) [for
                  multiclass case] as the probability measure returned by the model.
     C2. init_Xavier_weights()
         Assign weights with a Xavier uniform (or orthogonal) distribution and bias as 0.
         See https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
     """
    def __init__(self,
                 n_feat: int,
                 n_units: int,
                 n_classes: int,
                 n_layers: int = 2,
                 bidirectional: bool = False):
        super().__init__()

        # Type and value check
        assert isinstance(n_feat, int), \
            f"n_feat must be a positive integer. Now its type is {type(n_feat)}."
        assert n_feat >= 1, \
            f"n_feat must be a positive integer. Now its value is {n_feat}."
        self.n_feat = n_feat
        assert isinstance(n_units, int), \
            f"n_units must be a positive integer. Now its type is {type(n_units)}."
        assert n_units >= 1, \
            f"n_units must be a positive integer. Now its value is {n_units}."
        self.n_units = n_units
        assert isinstance(n_classes, int), \
            f"n_classes must be a positive integer. Now its type is {type(n_classes)}."
        assert n_classes >= 2, \
            f"n_classes must be a positive integer not less than 2. Now its value is {n_classes}."
        self.n_classes = n_classes
        assert isinstance(n_layers, int), \
            f"n_layers must be a positive integer. Now its type is {type(n_layers)}."
        assert n_layers >= 1, \
            f"n_layers must be a positive integer. Now its value is {n_layers}."
        self.n_layers = n_layers
        assert isinstance(bidirectional, bool), \
            f"bidirectional must be a boolean. Now its type is {type(bidirectional)}."
        self.bidirectional = bidirectional

        # Create model
        self.model = nn.LSTM(input_size=self.n_feat, num_layers=self.n_layers, hidden_size=self.n_units,
                             bidirectional=self.bidirectional, batch_first=True)
        # The layer before the output layer has doubled number of hidden units if bidirectional
        self.output_layer = nn.Linear(self.n_units * 2 if self.bidirectional else self.n_units,
                                      1 if self.n_classes == 2 else self.n_classes)

    def forward(self, X: Union[np.ndarray, torch.Tensor]):
        try:
            X = torch.Tensor(X)
        except:
            raise TypeError(f'X must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert len(X.shape) == 3, \
            f'X must be 3-dimensional. Now it is {len(X.shape)}-dimensional.'
        device = self.get_device()
        X = X.to(device)
        out, (h, c) = self.model(X)
        # https://discuss.pytorch.org/t/h-0-and-c-0-inputs-for-lstms-clarification/186281/2
        # h: all hidden states of all hidden layers in the last timestamp
        # out: all hidden states in the last one or two layers of all time timestamps, depending on the bidirectional
        # Approach 1: get torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) if bidirectional else h[-1, :, :]
        # Approach 2: get out[:, -1, :] to get the last timestamp result
        # Two approaches are identical.
        result = self.output_layer(out[:, -1, :])
        result_prob = torch.reshape(torch.sigmoid(result), shape=(-1,)) if self.n_classes == 2 \
            else torch.nn.functional.softmax(result, dim=1)   # sigmoid (softmax) for binary (multiclass) classification
        return result_prob

    def init_Xavier_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
              torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
              param.data.fill_(0)

########################################################################################################################
# Define an LSTM model class for regression tasks
# References
# https://www.kaggle.com/code/khalildmk/simple-two-layer-bidirectional-lstm-with-pytorch
# https://medium.com/@reddyyashu20/bidirectional-rnn-python-code-in-keras-and-pytorch-22b9a9a3c034
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_bidirectional_lstm.py
########################################################################################################################

class LSTM_Regressor(DL_Class):
    """
     A PyTorch LSTM class for regression.

     A. Runtime parameters
     ---------------------
     A1. n_feat: A positive integer.
         The number of features to be fitted to the model.
     A2. n_units: A positive integer.
         The number of hidden units in each hidden layer.
     A3. n_layers: A positive integer.
         The number of hidden layers in the model.
         Default setting: n_layers=2
     A4. bidirectional: A boolean.
         Whether the model will be constructed with a bidirectional layer for each hidden layer.
         Default setting: bidirectional=False

     B. Attributes
     -------------
     B1. model: A torch.nn.modules.container.Sequential object.
         Together with the output_layer in B2, it functions as the actual model .
     B2. output_layer: A torch.nn.Linear object.
         The output layer of the actual model.
     (A1-A4 are initialized as instance attributes.)

     C. Methods
     ----------
     C1. forward(X)
         :param X: A 3-dimensional numpy array (or Torch.Tensor) with dimension as (sample size, number of timestamps,
                   number of features)
         :return: A torch.Tensor of dimension (samples size, ) as the model output.
     C2. init_Xavier_weights()
         Assign weights with a Xavier uniform (or orthogonal) distribution and bias as 0.
         See https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
     """
    def __init__(self,
                 n_feat: int,
                 n_units: int,
                 n_layers: int = 2,
                 bidirectional: bool = False):
        super().__init__()

        # Type and value check
        assert isinstance(n_feat, int), \
            f"n_feat must be a positive integer. Now its type is {type(n_feat)}."
        assert n_feat >= 1, \
            f"n_feat must be a positive integer. Now its value is {n_feat}."
        self.n_feat = n_feat
        assert isinstance(n_units, int), \
            f"n_units must be a positive integer. Now its type is {type(n_units)}."
        assert n_units >= 1, \
            f"n_units must be a positive integer. Now its value is {n_units}."
        self.n_units = n_units
        assert isinstance(n_layers, int), \
            f"n_layers must be a positive integer. Now its type is {type(n_layers)}."
        assert n_layers >= 1, \
            f"n_layers must be a positive integer. Now its value is {n_layers}."
        self.n_layers = n_layers
        assert isinstance(bidirectional, bool), \
            f"bidirectional must be a boolean. Now its type is {type(bidirectional)}."
        self.bidirectional = bidirectional

        # Create model
        self.model = nn.LSTM(input_size=self.n_feat, num_layers=self.n_layers, hidden_size=self.n_units,
                             bidirectional=self.bidirectional, batch_first=True)
        # The layer before the output layer has doubled number of hidden units if bidirectional
        self.output_layer = nn.Linear(self.n_units * 2 if self.bidirectional else self.n_units, 1)

    def forward(self, X: Union[np.ndarray, torch.Tensor]):
        try:
            X = torch.Tensor(X)
        except:
            raise TypeError(f'X must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert len(X.shape) == 3, \
            f'X must be 3-dimensional. Now it is {len(X.shape)}-dimensional.'
        device = self.get_device()
        X = X.to(device)
        out, (h, c) = self.model(X)
        # https://discuss.pytorch.org/t/h-0-and-c-0-inputs-for-lstms-clarification/186281/2
        # h: all hidden states of all hidden layers in the last timestamp
        # out: all hidden states in the last one or two layers of all time timestamps, depending on the bidirectional
        # Approach 1: get torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) if bidirectional else h[-1, :, :]
        # Approach 2: get out[:, -1, :] to get the last timestamp result
        # Two approaches are identical.
        result = self.output_layer(out[:, -1, :])
        return torch.reshape(result, (-1,))

    def init_Xavier_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
              torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
              param.data.fill_(0)

########################################################################################################################
# Define a Transformer model class for classification
# References
# https://aravindkolli.medium.com/mastering-tabular-data-with-tabtransformer-a-comprehensive-guide-119f6dbf5a79
########################################################################################################################

class Transformer_Classifier(DL_Class):
    """
    A PyTorch Transformer class for classification.

    A. Runtime parameters
    ---------------------
    A1. n_feat: A positive integer.
        The number of features to be fitted to the model.
    A2. n_timestamps: A positive integer.
        The number of timestamps to be fitted to the model.
    A3. d_model: A positive integer.
        The dimensionality of the embedding vector for each timestamp.
    A4. n_classes: A positive integer greater than 1.
        The number of classes in the target.
    A5. n_layers: A positive integer.
        The number of hidden layers in the Transformer encoder.
        Default setting: n_layers=2
    A6. n_units: A positive integer.
        The number of hidden units in each hidden layer.
        Default setting: n_units=1024
    A7. n_heads: A positive integer.
        The number of attention heads in the Transformer.
        Default setting: n_heads=8

    B. Attributes
    -------------
    B1. embedding: A torch.nn.Linear object.
        The embedding layer.
    B2. positional_encoding: A torch.nn.Parameter object.
        A learnable parameter object to determine the time-order
    B3. transformer_encoder: A torch.nn.TransformerEncoder object.
        The encoding layer.
    B4. fc: A torch.nn.Linear object.
        A fully connected layer.
    (A1-A7 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. forward(X)
        :param X: A 3-dimensional numpy array (or Torch.Tensor) with dimension as (sample size, number of timestamps,
                  number of features)
        :return: A torch.Tensor of dimension (samples size, ) [for binary case] or (sample size, n_classes) [for
                 multiclass case] as the probability measure returned by the model.
    C2. init_Xavier_weights()
        Assign weights with a Xavier uniform (or orthogonal) distribution and bias as 0.
        See https://paperswithcode.com/method/xavier-initialization
    """
    def __init__(self,
                 n_feat: int,
                 n_timestamps: int,
                 d_model: int,
                 n_classes: int,
                 n_layers: int = 2,
                 n_units: int = 1024,
                 n_heads: int = 8):
        super().__init__()

        # Type and value check
        assert isinstance(n_feat, int), \
            f"n_feat must be a positive integer. Now its type is {type(n_feat)}."
        assert n_feat >= 1, \
            f"n_feat must be a positive integer. Now its value is {n_feat}."
        self.n_feat = n_feat
        assert isinstance(n_timestamps, int), \
            f"n_timestamps must be a positive integer. Now its type is {type(n_timestamps)}."
        assert n_feat >= 1, \
            f"n_timestamps must be a positive integer. Now its value is {n_timestamps}."
        self.n_timestamps = n_timestamps
        assert isinstance(d_model, int), \
            f"d_model must be a positive integer. Now its type is {type(d_model)}."
        assert d_model >= 1, \
            f"d_model must be a positive integer. Now its value is {d_model}."
        self.d_model = d_model
        assert isinstance(n_classes, int), \
            f"n_classes must be a positive integer. Now its type is {type(n_classes)}."
        assert n_classes >= 2, \
            f"n_classes must be a positive integer not less than 2. Now its value is {n_classes}."
        self.n_classes = n_classes
        assert isinstance(n_layers, int), \
            f"n_layers must be a positive integer. Now its type is {type(n_layers)}."
        assert n_layers >= 1, \
            f"n_layers must be a positive integer. Now its value is {n_layers}."
        self.n_layers = n_layers
        assert isinstance(n_units, int), \
            f"n_units must be a positive integer. Now its type is {type(n_units)}."
        assert n_units >= 1, \
            f"n_units must be a positive integer. Now its value is {n_units}."
        self.n_units = n_units
        assert isinstance(n_heads, int), \
            f"n_heads must be a positive integer. Now its type is {type(n_heads)}."
        assert n_heads >= 1, \
            f"n_heads must be a positive integer. Now its value is {n_heads}."
        self.n_heads = n_heads
        assert d_model % n_heads == 0, \
            f'd_model (={d_model}) must be divisible by n_heads (={n_heads}).'

        # Define layers
        self.embedding = nn.Linear(n_feat, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, n_timestamps, d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=n_units, batch_first=True),
            num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_classes if n_classes > 2 else 1)

    def forward(self, X):
        try:
            X = torch.Tensor(X)
        except:
            raise TypeError(f'X must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert len(X.shape) == 3, \
            f'X must be 3-dimensional. Now it is {len(X.shape)}-dimensional.'
        device = self.get_device()
        X = X.to(device)
        X = self.embedding(X)            # Step 1: Project the features to a {d_model}-dimensional embedding vector space
        X += self.positional_encoding    # Step 2: Add the relative positions of timestamps as a sequence of parameters
        X = self.transformer_encoder(X)  # Step 3: Run the encoder in a transformer
        X = X.mean(dim=1)                # Step 4: Global averaging pooling layer
        X = self.fc(X)                   # Step 5: Fully connected layer right before classification
        y = torch.reshape(torch.sigmoid(X), shape=(-1,)) if self.n_classes == 2 else torch.nn.functional.softmax(X, dim=1)
        return y                         # Step 6: Output layer for classification

    def init_Xavier_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name or 'positional_encoding' in name:
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param.data)
                elif param.dim() == 1:
                    param.data.fill_(1)
            elif 'bias' in name:
                param.data.fill_(0)

########################################################################################################################
# Define a Transformer model class for regression
# References
# https://aravindkolli.medium.com/mastering-tabular-data-with-tabtransformer-a-comprehensive-guide-119f6dbf5a79
########################################################################################################################

class Transformer_Regressor(DL_Class):
    """
    A PyTorch Transformer class for regression.

    A. Runtime parameters
    ---------------------
    A1. n_feat: A positive integer.
        The number of features to be fitted to the model.
    A2. n_timestamps: A positive integer.
        The number of timestamps to be fitted to the model.
    A3. d_model: A positive integer.
        The dimensionality of the embedding vector for each timestamp.
    A4. n_layers: A positive integer.
        The number of hidden layers in the Transformer encoder.
        Default setting: n_layers=2
    A5. n_units: A positive integer.
        The number of hidden units in each hidden layer.
        Default setting: n_units=1024
    A6. n_heads: A positive integer.
        The number of attention heads in the Transformer.
        Default setting: n_heads=8

    B. Attributes
    -------------
    B1. embedding: A torch.nn.Linear object.
        The embedding layer.
    B2. positional_encoding: A torch.nn.Parameter object.
        A learnable parameter object to determine the time-order
    B3. transformer_encoder: A torch.nn.TransformerEncoder object.
        The encoding layer.
    B4. fc: A torch.nn.Linear object.
        A fully connected layer.
    (A1-A6 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. forward(X)
        :param X: A 3-dimensional numpy array (or Torch.Tensor) with dimension as (sample size, number of timestamps,
                  number of features)
        :return: A torch.Tensor of dimension (samples size, ) as the model output.
    C2. init_Xavier_weights()
        Assign weights with a Xavier uniform (or orthogonal) distribution and bias as 0.
        See https://paperswithcode.com/method/xavier-initialization
    """
    def __init__(self,
                 n_feat: int,
                 n_timestamps: int,
                 d_model: int,
                 n_layers: int = 2,
                 n_units: int = 1024,
                 n_heads: int = 8):
        super().__init__()

        # Type and value check
        assert isinstance(n_feat, int), \
            f"n_feat must be a positive integer. Now its type is {type(n_feat)}."
        assert n_feat >= 1, \
            f"n_feat must be a positive integer. Now its value is {n_feat}."
        self.n_feat = n_feat
        assert isinstance(n_timestamps, int), \
            f"n_timestamps must be a positive integer. Now its type is {type(n_timestamps)}."
        assert n_feat >= 1, \
            f"n_timestamps must be a positive integer. Now its value is {n_timestamps}."
        self.n_timestamps = n_timestamps
        assert isinstance(d_model, int), \
            f"d_model must be a positive integer. Now its type is {type(d_model)}."
        assert d_model >= 1, \
            f"d_model must be a positive integer. Now its value is {d_model}."
        self.d_model = d_model
        assert isinstance(n_layers, int), \
            f"n_layers must be a positive integer. Now its type is {type(n_layers)}."
        assert n_layers >= 1, \
            f"n_layers must be a positive integer. Now its value is {n_layers}."
        self.n_layers = n_layers
        assert isinstance(n_units, int), \
            f"n_units must be a positive integer. Now its type is {type(n_units)}."
        assert n_units >= 1, \
            f"n_units must be a positive integer. Now its value is {n_units}."
        self.n_units = n_units
        assert isinstance(n_heads, int), \
            f"n_heads must be a positive integer. Now its type is {type(n_heads)}."
        assert n_heads >= 1, \
            f"n_heads must be a positive integer. Now its value is {n_heads}."
        self.n_heads = n_heads
        assert n_units % d_model == 0, \
            f'n_units (={n_units}) must be divisible by d_model (={d_model}).'

        # Define layers
        self.embedding = nn.Linear(n_feat, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, n_timestamps, d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=n_units, batch_first=True),
            num_layers=n_layers)
        self.fc = nn.Linear(d_model,1)

    def forward(self, X):
        try:
            X = torch.Tensor(X)
        except:
            raise TypeError(f'X must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert len(X.shape) == 3, \
            f'X must be 3-dimensional. Now it is {len(X.shape)}-dimensional.'
        device = self.get_device()
        X = X.to(device)

        X = self.embedding(X)            # Step 1: Project the features to a {d_model}-dimensional embedding vector space
        X += self.positional_encoding    # Step 2: Add the relative positions of timestamps as a sequence of parameters
        X = self.transformer_encoder(X)  # Step 3: Run the encoder in a transformer
        X = X.mean(dim=1)                # Step 4: Global averaging pooling layer
        y = self.fc(X).squeeze()         # Step 5: Fully connected layer for regression
        return y

    def init_Xavier_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name or 'positional_encoding' in name:
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param.data)
                elif param.dim() == 1:
                    param.data.fill_(1)
            elif 'bias' in name:
                param.data.fill_(0)

########################################################################################################################
# Define a Temporal Block for Temporal Convolutional Network (TCN) model class
# References
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
########################################################################################################################

class TemporalBlock(nn.Module):
    """
    A PyTorch temporal block class for one layer in a TCN model.

    A. Runtime parameters
    ---------------------
    A1. in_channels: A positive integer.
        The number of input channels (e.g., number of features to be fitted in the first layer of a TCN model).
    A2. out_channels: A positive integer
        The number of output channels (i.e., filters).
    A3. kernel_size: A positive integer.
        The size of the convolutional kernel.
    A4. stride: A positive integer.
        The size of the stride on the convolutional layers.
    A5. dilation: A positive integer.
        The dilation factor for the convolutional kernel, which increases the receptive field without increasing the
        number of model parameters.

    B. Attributes
    -------------
    B1. model: A torch.nn.modules.container.Sequential object.
        Together with the down-sampling layer in B3, it functions as the actual model.
    B2. downsampling: a torch.nn.Conv1d object or a torch.nn.Identity() object.
        A layer used to align the input and output channels if needed.

    C. Methods
    ----------
    C1. forward(X)
        :param X: A 3-dimensional numpy array (or Torch.Tensor) with dimension as (sample size, number of timestamps,
                  number of features)
        :return: A torch.Tensor of dimension (samples size, ) [for binary case] or (sample size, n_classes) [for
                 multiclass case] as the probability measure returned by the model.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int):
        super(TemporalBlock, self).__init__()

        # Type and value check
        for arg, arg_name in zip([in_channels, out_channels, kernel_size, stride, dilation],
                                 ['in_channels', 'out_channels', 'kernel_size', 'stride', 'dilation']):
            assert isinstance(arg, int), \
                f'{arg} must be a positive integer. Now its type is {type(arg)}.'
            assert arg >= 1, \
                f'{arg} must be a positive integer. Now it is {arg}.'

        # Create a sequential model
        modules = []
        for i in range(2):
            padding = (kernel_size - 1) * dilation // 2       # padding that ensures the input and output lengths match
            conv_layer = nn.Conv1d(in_channels=in_channels if i == 0 else out_channels,
                                   out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation)  # 1-dimensional convolutional layer
            modules += [conv_layer, nn.ReLU()]
        self.model = nn.Sequential(*modules)    # Conventionally termed as 1 layer (despite 2 convolution layers)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels \
            else nn.Identity()                  # Align the input and output channels if needed

    def forward(self, X):
        out, residual = self.model(X), self.downsample(X)
        return nn.ReLU()(out + residual)

########################################################################################################################
# Define a Temporal Convolutional Network (TCN) model class for classification
# References
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
########################################################################################################################

class TCN_Classifier(DL_Class):
    """
    A PyTorch TCN class for classification.

    A. Runtime parameters
    ---------------------
    A1. n_feat: A positive integer.
        The number of features to be fitted to the model.
    A2. n_units: A positive integer.
        The number of out_channels in each hidden layer (= TemporalBlock object)
    A3. n_classes: A positive integer greater than 1.
        The number of classes in the target.
    A4. n_layers: A positive integer.
        The number of hidden layers (i.e., the number of TemporalBlock objects embedded).
        Default setting: n_layers=2
    A5. kernel_size: A positive integer.
        The size of the convolutional kernel.
        Default setting: kernel_size=3

    B. Attributes
    -------------
    B1. model: A torch.nn.modules.container.Sequential object.
        Together with the fully connected layer in B2, it functions as the actual model.
    B2. fc: A torch.nn.Linear object.
        A fully connected layer.
    (A1-A5 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. forward(X)
        :param X: A 3-dimensional numpy array (or Torch.Tensor) with dimension as (sample size, number of timestamps,
                   number of features)
        :return: A torch.Tensor of dimension (samples size, ) [for binary case] or (sample size, n_classes) [for
                  multiclass case] as the probability measure returned by the model.
    C2. init_Xavier_weights()
        Assign weights with a Xavier uniform distribution and bias as 0.
        See https://paperswithcode.com/method/xavier-initialization
    """

    def __init__(self,
                 n_feat: int,
                 n_units: int,
                 n_classes: int,
                 n_layers: int = 2,
                 kernel_size: int = 3):
        super().__init__()

        # Type and value check
        for arg, arg_name in zip([n_feat, n_units, n_layers, kernel_size],
                               ['n_feat', 'n_units', 'n_layers', 'kernel_size']):
            assert isinstance(arg, int), \
                f'{arg} must be a positive integer. Now its type is {type(arg)}.'
            assert arg >= 1, \
                f'{arg} must be a positive integer. Now it is {arg}.'
        self.n_feat = n_feat
        self.n_units = n_units
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        assert isinstance(n_classes, int), \
            f"n_classes must be a positive integer. Now its type is {type(n_classes)}."
        assert n_classes >= 2, \
            f"n_classes must be a positive integer not less than 2. Now its value is {n_classes}."
        self.n_classes = n_classes

        # Create a sequential model
        modules = []
        for i in range(n_layers):
            dilation_size = 2 ** i      # Increasing dilation size exponentially by layers: 1, 2, 4, ...
            in_channels = n_feat if i == 0 else n_units
            out_channels = n_units
            modules.append(TemporalBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=1,                      # Fix the stride as 1 following the common
                                         dilation=dilation_size))       # practice for tabular data
        self.model = nn.Sequential(*modules)
        self.fc = nn.Linear(n_units, n_classes if n_classes > 2 else 1)

    def forward(self, X):
        try:
          X = torch.Tensor(X)
        except:
          raise TypeError(f'X must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert len(X.shape) == 3, \
          f'X must be 3-dimensional. Now it is {len(X.shape)}-dimensional.'
        device = self.get_device()
        X = X.to(device)
        X = X.permute(0, 2, 1)    # batch-first approach
        X = self.model(X)
        result = self.fc(X[:, :, -1])   # Output from the last timestamp
        result_prob = torch.sigmoid(result).squeeze(-1) if self.n_classes == 2 \
          else torch.nn.functional.softmax(result, dim=1)  # sigmoid (softmax) for binary (multiclass) classification
        return result_prob

    def init_Xavier_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
             torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
              param.data.fill_(0)

########################################################################################################################
# Define a Temporal Convolutional Network (TCN) model class for classification
# References
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
########################################################################################################################

class TCN_Regressor(DL_Class):
    """
    A PyTorch TCN class for regression.

    A. Runtime parameters
    ---------------------
    A1. n_feat: A positive integer.
        The number of features to be fitted to the model.
    A2. n_units: A positive integer.
        The number of out_channels in each hidden layer (= TemporalBlock object)
    A3. n_layers: A positive integer.
        The number of hidden layers (i.e., the number of TemporalBlock objects embedded).
        Default setting: n_layers=2
    A4. kernel_size: A positive integer.
        The size of the convolutional kernel.
        Default setting: kernel_size=3

    B. Attributes
    -------------
    B1. model: A torch.nn.modules.container.Sequential object.
        Together with the fully connected layer in B2, it functions as the actual model.
    B2. fc: A torch.nn.Linear object.
        A fully connected layer.
    (A1-A4 are initialized as instance attributes.)

    C. Methods
    ----------
    C1. forward(X)
        :param X: A 3-dimensional numpy array (or Torch.Tensor) with dimension as (sample size, number of timestamps,
                   number of features)
        :return: A torch.Tensor of dimension (samples size, ) [for binary case] or (sample size, n_classes) [for
                  multiclass case] as the probability measure returned by the model.
    C2. init_Xavier_weights()
        Assign weights with a Xavier uniform distribution and bias as 0.
        See https://paperswithcode.com/method/xavier-initialization
    """

    def __init__(self,
               n_feat: int,
               n_units: int,
               n_layers: int = 2,
               kernel_size: int = 3):
        super().__init__()

        # Type and value check
        for arg, arg_name in zip([n_feat, n_units, n_layers, kernel_size],
                               ['n_feat', 'n_units', 'n_layers', 'kernel_size']):
            assert isinstance(arg, int), \
                f'{arg} must be a positive integer. Now its type is {type(arg)}.'
            assert arg >= 1, \
                f'{arg} must be a positive integer. Now it is {arg}.'
        self.n_feat = n_feat
        self.n_units = n_units
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # Create a sequential model
        modules = []
        for i in range(n_layers):
            dilation_size = 2 ** i      # Increasing dilation size exponentially by layers: 1, 2, 4, ...
            in_channels = n_feat if i == 0 else n_units
            out_channels = n_units
            modules.append(TemporalBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=1,                      # Fix the stride as 1 following the common
                                         dilation=dilation_size))       # practice for tabular data
        self.model = nn.Sequential(*modules)
        self.fc = nn.Linear(n_units, 1)

    def forward(self, X):
        try:
          X = torch.Tensor(X)
        except:
          raise TypeError(f'X must be (convertible to) a torch.Tensor. Now its type is {type(X)}.')
        assert len(X.shape) == 3, \
          f'X must be 3-dimensional. Now it is {len(X.shape)}-dimensional.'
        device = self.get_device()
        X = X.to(device)
        X = X.permute(0, 2, 1)    # batch-first approach
        X = self.model(X)
        result = self.fc(X[:, :, -1]).squeeze()   # Output from the last timestamp
        return result

    def init_Xavier_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
             torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
              param.data.fill_(0)

########################################################################################################################
# Define the function to train a model
########################################################################################################################

def train_model(model: Union[ANN_Classifier, ANN_Regressor, LSTM_Classifier, LSTM_Regressor,
                             Transformer_Classifier, Transformer_Regressor, TCN_Classifier, TCN_Regressor],
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                X_val: torch.Tensor,
                y_val: torch.Tensor,
                n_epochs: int,
                criterion: LossFunction,
                optimizer: torch.optim.Optimizer,
                earlyStopper: Optional[EarlyStopping] = None,
                verbose_epoch: Optional[int] = None,
                **kwargs):
    """
    :param model: An object of any class in the list: [ANN_Classifier, ANN_Regressor, LSTM_Classifier, LSTM_Regressor,
                  Transformer_Classifier, Transformer_Regressor, TCN_Classifier, TCN_Regressor].
    :param X_train: A 2- or 3-dimensional numpy array (or Torch.Tensor).
           The training feature data with dimension of (sample size, number of features) if model is an ANN object, or
           (sample size, number of timestamps, number of features) if LSTM model is an LSTM object.
    :param y_train: A 1-dimensional numpy array (or Torch.Tensor).
           The training target 1-dimensional data with length as the sample size.
    :param X_val: A 2- or 3-dimensional numpy array (or Torch.Tensor).
           The validation feature data with dimension of (sample size, number of features) if model is an ANN
           object, or (sample size, number of timestamps, number of features) if LSTM model is an LSTM object.
    :param y_val: A 1-dimensional numpy array (or Torch.Tensor).
           The validation target 1-dimensional data with length as the sample size.
    :param n_epochs: A positive integer.
           The number of maximum epochs to be run in training the model.
    :param criterion: A loss function from torch.nn.
           See https://pytorch.org/docs/main/nn.html#loss-functions. Make sure you specify it WITH brackets.
           Example: torch.nn.CrossEntropyLoss()
    :param optimizer: An torch.optim.Optimizer object.
           See https://pytorch.org/docs/stable/optim.html. Make sure you specify it WITHOUT brackets.
           Example: torch.nn.AdamW
    :param earlyStopper: An EarlyStopping object or None.
           It aims to speed up training time and prevent over-fitting.
           Default setting: earlyStopper=None
    :param verbose_epoch: A positive integer or None.
           If integer, it controls the frequency of printing the training and validation losses with a rate of every
           {verbose_epoch} epochs. No logging will be printed if None.
           Default setting: verbose_epoch=None
    :param kwargs: (Any extra runtime parameters of optimizer)
           Example: lr=0.001 for the learning rate parameter of the optimizer.
    :return:
    A dictionary of the following four pairs of items:
    - 'Elapsed_train_time': the elapsed time of training the model
    - 'Elapsed_train_epochs': the elapsed epochs of training the model
    - 'Train_loss': the training loss
    - 'Val_loss': the validation loss
    """

    # Type and value check
    assert type(model) in [ANN_Classifier, ANN_Regressor, LSTM_Classifier, LSTM_Regressor,
                           Transformer_Classifier, Transformer_Regressor, TCN_Classifier, TCN_Regressor], \
        (f'model must be an object from [ANN_Classifier, ANN_Regressor, LSTM_Classifier, LSTM_Regressor, '
         f'Transformer_Classifier, Transformer_Regressor, TCN_Classifier, TCN_Regressor]. Now its type is {type(model)}.')
    device = model.get_device()
    try:
        X_train = torch.Tensor(X_train).to(device)
    except:
        raise TypeError(f'X_train must be (convertible to) a torch.tensor. Now its type is {type(X_train)}.')
    try:
        y_train = torch.Tensor(y_train).to(device)
    except:
        raise TypeError(f'y_train must be (convertible to) a torch.tensor. Now its type is {type(y_train)}.')
    try:
        X_val = torch.Tensor(X_val).to(device)
    except:
        raise TypeError(f'X_val must be (convertible to) a torch.tensor. Now its type is {type(X_val)}.')
    try:
        y_val = torch.Tensor(y_val).to(device)
    except:
        raise TypeError(f'y_val must be (convertible to) a torch.tensor. Now its type is {type(y_val)}.')
    assert len(X_train.shape) in [2, 3], \
        f'X_train must be two- or three-dimensional. Now its dimension is {X_train.shape}'
    assert len(X_val.shape) in [2, 3], \
        f'X_val must be two- or three-dimensional. Now its dimension is {X_val.shape}'
    assert len(y_train.shape) == 1, \
        f'y_train must be one-dimensional. Now its dimension is {y_train.shape}.'
    assert len(y_val.shape) == 1, \
        f'y_val must be one-dimensional. Now its dimension is {y_val.shape}.'
    assert isinstance(n_epochs, int), \
        f"n_epochs must be a positive integer. Now its type is {type(n_epochs)}."
    assert n_epochs >= 1, \
        f"n_epochs must be a positive integer. Now it is {n_epochs}."
    # No type check for criterion and optimizer because PyTorch did not define the associated class.
    if earlyStopper is not None:
        assert isinstance(earlyStopper, EarlyStopping), \
            f"earlyStopper (if not None) must be a EarlyStopping object. Now its type is {type(earlyStopper)}."
    if verbose_epoch is not None:
        assert isinstance(verbose_epoch, int), \
            f"verbose_epoch (if not None) must be a positive integer. Now its type is {type(verbose_epoch)}."
        assert n_epochs >= verbose_epoch >= 1, \
            f"verbose_epoch must be in the range [1, n_epochs]. Now its value is {n_epochs}."

    # Specify the optimizer using keyword arguments
    opt = optimizer(model.parameters(), **kwargs)

    # Create dummy loss values and epoch counter
    train_loss, val_loss, elapsed_epochs = None, None, 0

    train_start = time()        # Training starts here
    for epoch_idx in range(n_epochs):
        model.train()
        opt.zero_grad()
        y_train_pred = model(X_train)
        try:
            train_loss = criterion(y_train_pred, y_train)
        except RuntimeError:
            train_loss = criterion(y_train_pred, y_train.long())    # CrossEntropyLoss and NLLLoss require LongTensor

        # Backward propagation
        train_loss.backward()
        opt.step()
        model.eval()
        elapsed_epochs += 1

        # Validation
        with torch.no_grad():
            y_val_pred = model(X_val)
            try:
                val_loss = criterion(y_val_pred, y_val)
            except RuntimeError:
                val_loss = criterion(y_val_pred, y_val.long())      # CrossEntropyLoss and NLLLoss require LongTensor
            if verbose_epoch is not None and (epoch_idx+1) % verbose_epoch == 0:
                print(f"Epochs: {epoch_idx+1:4}/{n_epochs:>4}; Training loss = {train_loss:.4f}; "
                      f"Validation loss = {val_loss:.4f}", flush=True)
        if earlyStopper is not None:
            if earlyStopper.early_stop(val_loss):
                break

    train_end = time()      # Training ends here
    return {'Elapsed_train_time': train_end - train_start, 'Elapsed_train_epochs': elapsed_epochs,
            'Train_loss': train_loss.item(), 'Val_loss': val_loss.item()}

########################################################################################################################

def test_model(model: Union[ANN_Classifier, ANN_Regressor, LSTM_Classifier, LSTM_Regressor,
                            Transformer_Classifier, Transformer_Regressor, TCN_Classifier, TCN_Regressor],
               X: torch.Tensor,
               y: torch.Tensor,
               criterion: LossFunction,
               prefix: str = 'Test_',
               return_pred: bool = False):
    """
    :param model: An object of any class in the list: [ANN_Classifier, ANN_Regressor, LSTM_Classifier, LSTM_Regressor,
                  Transformer_Classifier, Transformer_Regressor, TCN_Classifier, TCN_Regressor].
    :param X: A 2- or 3-dimensional numpy array (or Torch.Tensor).
           The test feature data with dimension of (sample size, number of features) if model is an ANN object, or
           (sample size, number of timestamps, number of features) if LSTM model is an LSTM object.
    :param y:  A 1-dimensional numpy array (or Torch.Tensor).
           The test target 1-dimensional data with length as the sample size.
    :param criterion: A loss function from torch.nn.
           See https://pytorch.org/docs/main/nn.html#loss-functions. Make sure you specify it WITH brackets.
           Example: torch.nn.CrossEntropyLoss()
    :param prefix: A string.
           The prefix (e.g. 'Test_') of the metric name shown in the keys of the output dictionary.
           Default setting: prefix='Test_'
    :param return_pred: A boolean.
           Return the predicted labels as well if True, or only the loss (as a dictionary) otherwise.
           Default setting: return_pred=False
    :return:
    (a) A dictionary with key as '{prefix}loss' and value as the loss value.
    If return_pred=True:
    (b) The predicted labels or probability measures as a torch.tensor.
    """

    # Type and value check
    assert type(model) in [ANN_Classifier, ANN_Regressor, LSTM_Classifier, LSTM_Regressor,
                           Transformer_Classifier, Transformer_Regressor, TCN_Classifier, TCN_Regressor], \
        (f'model must be an object from [ANN_Classifier, ANN_Regressor, LSTM_Classifier, LSTM_Regressor, '
         f'Transformer_Classifier, Transformer_Regressor, TCN_Classifier, TCN_Regressor]. Now its type is {type(model)}.')
    model.eval()
    device = model.get_device()
    try:
        X = torch.Tensor(X).to(device)
    except:
        raise TypeError(f'X must be (convertible to) a torch.tensor. Now its type is {type(X)}.')
    try:
        y = torch.Tensor(y).to(device)
    except:
        raise TypeError(f'y must be (convertible to) a torch.tensor. Now its type is {type(y)}.')
    assert len(X.shape) in [2, 3], \
        f'X must be two- or three-dimensional. Now its dimension is {X.shape}'
    assert len(y.shape) == 1, \
        f'y must be one-dimensional. Now its dimension is {y.shape}.'
    assert isinstance(prefix, str), \
        f"prefix must be a string. Now its type is {type(prefix)}."
    assert isinstance(return_pred, bool), \
        f"return_pred must be a boolean. Now its type is {type(return_pred)}."

    with torch.no_grad():
        y_pred = model(X)
        try:
            test_loss = criterion(y_pred, y)
        except RuntimeError:
            test_loss = criterion(y_pred, y.long())        # CrossEntropyLoss and NLLLoss require LongTensor
    if return_pred:
        return {f'{prefix}loss': test_loss.item()}, y_pred
    else:
        return {f'{prefix}loss': test_loss.item()}

########################################################################################################################
