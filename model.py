from torch.nn import Module, LSTM, Linear, InstanceNorm1d, Parameter
import torch.nn.functional as F
import torch
import numpy as np


class OSU(Module):
    def __init__(
        self, nb_features, nb_frames,
        nb_channels=1, nb_layers=3, hidden_size=512,
        output_mean=None,
    ):
        super(OSU, self).__init__()

        self.hidden_size = hidden_size

        self.in0 = InstanceNorm1d(nb_features*nb_channels)

        self.fc1 = Linear(
            nb_features*nb_channels, hidden_size,
            bias=False
        )

        self.in1 = InstanceNorm1d(hidden_size)

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=nb_layers,
            bidirectional=True,
            batch_first=False,
        )

        self.fc2 = Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False
        )

        self.in2 = InstanceNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=nb_features*nb_channels,
            bias=False
        )

        self.in3 = InstanceNorm1d(hidden_size)

        self.output_scale = Parameter(
            torch.ones(nb_features).float()
        )

        self.output_mean = Parameter(
            torch.from_numpy(np.copy(output_mean)).float()
        )

    def forward(self, x):
        nb_frames, nb_batches, nb_channels, nb_features = x.data.shape
        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.reshape(nb_frames, nb_batches, nb_channels*nb_features)
        x = self.in0(x.permute(1, 2, 0)).permute(2, 0, 1)

        # reduce feature dimensions, therefore we reshape
        # to (nb_frames*nb_batches, nb_channels*nb_features)
        # and encode to (nb_frames*nb_batches, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*nb_features))
        x = x.reshape(nb_frames, nb_batches, self.hidden_size)
        # normalize every instance in a batch
        x = self.in1(x.permute(1, 2, 0)).permute(2, 0, 1)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # reshape to 1D vector (seq_len*batch, hidden_size)
        # add skip connection
        x = lstm_out[0]

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, self.hidden_size))
        x = x.reshape(nb_frames, nb_batches, self.hidden_size)
        x = self.in2(x.permute(1, 2, 0)).permute(2, 0, 1)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x.reshape(-1, self.hidden_size))
        x = x.reshape(nb_frames, nb_batches, nb_channels*nb_features)
        x = self.in3(x.permute(1, 2, 0)).permute(2, 0, 1)

        # reshape back to sequence
        x = x.reshape(nb_frames, nb_batches, nb_channels, nb_features)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x)

        return x
