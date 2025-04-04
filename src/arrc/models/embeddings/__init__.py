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

# Ready
from .simple_cnn import build_cnn_model
from .cnn_wavenet import build_cnn_wavenet_with_attentionpooling_model, build_cnn_wavenet_with_globalpooling_model
from .bayesian_cnn_lstm import build_bayesian_cnn_lstm_model
from .petsfcnn import build_petsfcnn_model
from .tcnn import build_tcnn_with_attentionpooling_model, build_tcnn_with_globalpooling_model
from .baseline import build_baseline_model
from .resnet import build_resnet_cnn_model
# Under Test:

# Pending Test
from arrc.models.embeddings.resnet import build_resnet_cnn_model
from arrc.models.embeddings.experimental.encoder import build_encoder_model
from arrc.models.embeddings.experimental.wavenet import build_wavenet_with_attention_model
from arrc.models.embeddings.experimental.dl_cnn_classification import build_dl_cnn_model
from arrc.models.embeddings.experimental.gated_resnet import build_gated_resnet
