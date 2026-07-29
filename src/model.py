"""
Convolutional Recurrent Neural Network (CRNN) Architecture for R.O.A.D. Barbados Historic Handwriting Challenge.

Modules:
- Module 1: CNN Feature Extractor (ResNet-18 backbone with height-only pooling to achieve 4x width downsampling)
- Module 2: RNN Sequence Modeler (2-layer Bidirectional LSTM)
- Module 3: CTC Linear Projection Head (Mapping to 76 vocabulary tokens)
- CTC Decoders: Greedy Search & Beam Search decoding utilities
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Dict, Any, Optional

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import Tokenizer


class ResNetCRNNBackbone(nn.Module):
    """
    Module 1: ResNet-18 Feature Extractor tailored for line crops (128px height).
    Width downsampling factor is strictly 4x (via conv1 & maxpool).
    Height is compressed from 128 -> 1 via (2,1) strides in layer2/3/4 and adaptive pooling.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Initial Conv & BatchNorm (Downsamples H & W by 2x each)
        self.conv1 = resnet.conv1  # stride (2, 2) -> H/2, W/2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride (2, 2) -> H/4, W/4

        # ResNet Layer 1 (64 channels, stride 1)
        self.layer1 = resnet.layer1

        # ResNet Layer 2 (128 channels): Modify stride to (2, 1) -> H/8, W/4
        self.layer2 = resnet.layer2
        self._modify_layer_stride(self.layer2, stride=(2, 1))

        # ResNet Layer 3 (256 channels): Modify stride to (2, 1) -> H/16, W/4
        self.layer3 = resnet.layer3
        self._modify_layer_stride(self.layer3, stride=(2, 1))

        # ResNet Layer 4 (512 channels): Modify stride to (2, 1) -> H/32, W/4
        self.layer4 = resnet.layer4
        self._modify_layer_stride(self.layer4, stride=(2, 1))

        # Adaptive pooling to strictly force height = 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

    def _modify_layer_stride(self, layer: nn.Sequential, stride: Tuple[int, int]):
        """Modifies first BasicBlock in a ResNet layer to use asymmetric stride (2, 1)."""
        layer[0].conv1.stride = stride
        if layer[0].downsample is not None:
            layer[0].downsample[0].stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, 3, 128, W)
        x = self.conv1(x)         # (Batch, 64, 64, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       # (Batch, 64, 32, W/4)

        x = self.layer1(x)        # (Batch, 64, 32, W/4)
        x = self.layer2(x)        # (Batch, 128, 16, W/4)
        x = self.layer3(x)        # (Batch, 256, 8, W/4)
        x = self.layer4(x)        # (Batch, 512, 4, W/4)

        x = self.adaptive_pool(x) # (Batch, 512, 1, W/4)
        return x


class CRNN_OCR(nn.Module):
    """
    Full CRNN Model: CNN Feature Extractor (4x width downsampling) + 2-layer BiLSTM + Linear CTC Head.
    """
    def __init__(
        self,
        vocab_size: int = 76,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
        pretrained_backbone: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Module 1: CNN Feature Extractor
        self.backbone = ResNetCRNNBackbone(pretrained=pretrained_backbone)

        # Module 2: Sequence Modeler (BiLSTM)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )

        # Module 3: CTC Linear Projection Head
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass Flow:
            Image (B, 3, 128, W) -> CNN -> (B, 512, 1, W/4) -> Squeeze/Permute ->
            BiLSTM -> (B, W/4, 512) -> FC -> Logits (B, W/4, Vocab_Size)
        """
        # Step 1: CNN Feature Extractor
        features = self.backbone(images)  # (Batch, 512, 1, W_seq)

        # Step 2: Squeeze Height dimension and Permute for RNN
        features = features.squeeze(2).permute(0, 2, 1)  # (Batch, W_seq, 512)

        # Step 3: BiLSTM Sequence Modeler
        rnn_out, _ = self.rnn(features)   # (Batch, W_seq, 512)

        # Step 4: CTC Linear Projection Head
        logits = self.fc(rnn_out)         # (Batch, W_seq, Vocab_Size)

        return logits

    def decode_greedy(self, logits: torch.Tensor, tokenizer: Tokenizer) -> List[str]:
        """
        Performs CTC Greedy Decoding:
            1. argmax over vocabulary dimension at each timestep
            2. Collapse consecutive CTC duplicate tokens
            3. Remove [BLANK] token (index 0)
        """
        probs = F.softmax(logits, dim=2)
        arg_maxes = torch.argmax(probs, dim=2)  # (Batch, W_seq)

        predictions = []
        for i in range(arg_maxes.size(0)):
            tokens = arg_maxes[i].tolist()
            decoded_text = tokenizer.decode(tokens, remove_ctc_duplicates=True)
            predictions.append(decoded_text)

        return predictions
