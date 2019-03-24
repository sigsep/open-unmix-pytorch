from torch.nn import LSTM, Linear, InstanceNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoOp(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class Spectrogram(nn.Module):
    def __init__(
        self, 
        n_fft=2048, 
        n_hop=1024, 
        power=1, 
        mono=True
    ):

        super(Spectrogram, self).__init__()

        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.power = power
        self.mono = mono

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesamples)
        Output:(nb_frames, nb_samples, nb_channels, nb_bins)
        """
        assert x.dim() == 3

        nb_samples, nb_channels, _ = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=True,
            normalized=False, onesided=True,
            pad_mode='reflect'
        ).transpose(1, 2)

        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)
        stft_f = stft_f.contiguous().view(nb_samples, nb_channels, -1, self.n_fft // 2 + 1)

        if self.mono:
            stft_f = torch.sqrt(torch.sum(stft_f**2, 1, keepdim=True))
        
        return stft_f.permute(2, 0, 1, 3)


class OSU(nn.Module):
    def __init__(
        self, 
        n_fft=2048,
        n_hop=1024,
        power=1,
        nb_channels=1, 
        nb_layers=3, 
        hidden_size=512,
        image=False
    ):
        super(OSU, self).__init__()

        self.hidden_size = hidden_size
        self.nb_bins = n_fft // 2 + 1
        if image:
            self.transform = NoOp()
        else:
            self.transform = Spectrogram(n_fft=n_fft, n_hop=n_hop, power=1, mono=(nb_channels == 1))
        self.in0 = InstanceNorm1d(self.nb_bins*nb_channels)

        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
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
            out_features=self.nb_bins*nb_channels,
            bias=False
        )

        self.in3 = InstanceNorm1d(hidden_size)

        self.fc4 = Linear(
            in_features=self.nb_bins*nb_channels,
            out_features=self.nb_bins*nb_channels,
            bias=True
        )

    def forward(self, x):
        # check for waveform or image
        # transform to spectrogram if (nb_batches, nb_channels, samples)
        x = self.transform(x)
        nb_frames, nb_batches, nb_channels, nb_bins = x.data.shape

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.reshape(nb_frames, nb_batches, nb_channels*nb_bins)
        x = self.in0(x.permute(1, 2, 0)).permute(2, 0, 1)

        # reduce feature dimensions, therefore we reshape
        # to (nb_frames*nb_batches, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_batches, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*nb_bins))
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
        x = x.reshape(nb_frames, nb_batches, nb_channels*nb_bins)
        x = self.in3(x.permute(1, 2, 0)).permute(2, 0, 1)
        
        # scale back to output domain
        x = self.fc4(x.reshape(-1, nb_channels*nb_bins))

        # reshape back to sequence
        x = x.reshape(nb_frames, nb_batches, nb_channels, nb_bins)

        # since our output is non-negative, we can apply RELU
        x = F.relu(x)
        
        return x
