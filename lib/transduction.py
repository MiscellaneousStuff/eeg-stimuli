import torch.nn as nn
import torch.nn.functional as F

from .transformer import *

class DigitalVoicingModel(nn.Module):
    def __init__(self,
                 ins,
                 model_size,
                 n_layers,
                 dropout,
                 outs):
        super().__init__()
        self.dropout = dropout
        self.lstm = \
            nn.LSTM(
                ins, model_size, batch_first=True,
                bidirectional=True, num_layers=n_layers,
                dropout=dropout)
        self.w1 = nn.Linear(model_size * 2, outs)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x, _ = self.lstm(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.w1(x)


class ProposedModel(nn.Module):
    def __init__(self,
                 model_size,
                 dropout=0.2,
                 num_layers=6,
                 n_heads=8,
                 dim_feedforward=3072,
                 out_dim=80,
                 use_resnet=False):
        super().__init__()
        self.use_resnet = use_resnet

        if use_resnet:
            self.conv_blocks = nn.Sequential(
                ResBlock(62, model_size, 2),
                ResBlock(model_size, model_size, 2),
                ResBlock(model_size, model_size, 2),
            )
            self.w_raw_in = nn.Linear(model_size, model_size)

        encoder_layer = TransformerEncoderLayer(
            d_model=model_size,
            nhead=n_heads,
            relative_positional=True,
            relative_positional_distance=100,
            dim_feedforward=dim_feedforward,
            dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.w_out = nn.Linear(model_size, out_dim)
    
    def forward(self, x):
        # (Optional) ResNet Blocks
        if self.use_resnet:
            # x shape is (batch, time, electrode)
            x = x.transpose(1, 2)
            x = self.conv_blocks(x)
            x = x.transpose(1, 2)
            x = self.w_raw_in(x)

        # Transformer
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        # Final MLP
        x = self.w_out(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)