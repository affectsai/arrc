#  Copyright (c) 2025 - Affects AI LLC
#
#  Licensed under the Creative Common CC BY-NC-SA 4.0 International License (the "License");
#  you may not use this file except in compliance with the License. The full text of the License is
#  provided in the included LICENSE file. If this file is not available, you may obtain a copy of the
#  License at
#
#       https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
#  Unless required by applicable law or agreed to in writing, software distributed under the License
#  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing permissions and limitations
#  under the License.

import arrc.layers
import keras.src.layers

from ..aer_model import ARRCModel
from arrc.layers import L2Normalization,NanCheckLayer
from .embedding_utils import *

def build_baseline_model(
        input_shape=(256 * 45, 1),  # e.g., 45s @ 256Hz , single channel
        dropout_rate=0.2,  # For GSO tuning in external code
        leaky_relu=True,  # For GSO tuning in external code,
        one_d_conv_1_filters=32,  #
        one_d_conv_2_filters=64,  #
        fc_layer_1_neurons=128,  # For GSO tuning in external code,
        fc_layer_2_neurons=64,  # For GSO tuning in external code,
        signal_duration=45,
        add_noise=True,
        random_shift=True,
        random_amp_scaling=True,
        **arrc_kwargs,
):
    """
    This is the 1-D CNN Branch of the PETSFCNN Model.

    Hammad, Dhiyaa Salih, and Hamed Monkaresi. "Ecg-based emotion detection via parallel-extraction of temporal and
    spatial features using convolutional neural network." Traitement du Signal 39.1 (2022): 43.
    """
    inputs = keras.layers.Input(shape=input_shape, name="1d_input")
    augmented = get_augmented_input(inputs, add_noise, random_shift, random_amp_scaling)

    # 1D CNN Branch
    one_d_block = keras.layers.Conv1D(filters=one_d_conv_1_filters, strides=1, kernel_size=5, name=f'1d_conv_1')(augmented)
    one_d_block = keras.layers.LeakyReLU()(one_d_block) if leaky_relu else keras.layers.ReLU()(one_d_block)
    one_d_block = keras.layers.MaxPool1D(pool_size=2, name=f'1d_pool_1')(one_d_block)
    one_d_block = keras.layers.Dropout(0.2)(one_d_block)
    one_d_block = keras.layers.Conv1D(one_d_conv_2_filters, kernel_size=3, name=f'1d_conv_2')(one_d_block)
    one_d_block = keras.layers.LeakyReLU()(one_d_block) if leaky_relu else keras.layers.ReLU()(one_d_block)
    one_d_block = keras.layers.MaxPool1D(pool_size=2, name=f'1d_pool_2')(one_d_block)
    one_d_block = keras.layers.Dropout(dropout_rate)(one_d_block)
    one_d_out = keras.layers.Flatten(name="1d_out")(one_d_block)


    fc_layer = keras.layers.Dense(fc_layer_1_neurons)(one_d_out)
    fc_layer = keras.layers.LeakyReLU()(fc_layer) if leaky_relu else keras.layers.ReLU()(fc_layer)
    fc_layer = keras.layers.BatchNormalization()(fc_layer)
    fc_layer = keras.layers.Dropout(dropout_rate)(fc_layer)
    fc_layer = keras.layers.Dense(fc_layer_2_neurons)(fc_layer)
    fc_layer = keras.layers.LeakyReLU()(fc_layer) if leaky_relu else keras.layers.ReLU()(fc_layer)
    fc_layer = keras.layers.BatchNormalization()(fc_layer)
    output = keras.layers.Dropout(dropout_rate)(fc_layer)
    output = fc_layer

    return ARRCModel.BuildARRCModel(
        inputs=inputs,
        embedding_outputs=output,
        name="Baseline",
        num_fc_layers=0,
        **arrc_kwargs,
    )
