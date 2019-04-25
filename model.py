from torch.nn import LSTM, Linear, InstanceNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        nb_samples, nb_channels, nb_bins, nb_frames, real_imag = stft_f.shape
        stft_f = stft_f.transpose(2, 3)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.sum(
                stft_f**2/self.power, 1, keepdim=True
            ) ** (self.power/2)

        # permute output for LSTM convenience
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
        image=False,
        output_mean=None,
        max_bin=None
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OSU, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))

        if image:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)

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
            dropout=0.5
        )

        self.fc2 = Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False
        )

        self.in2 = InstanceNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.in3 = InstanceNorm1d(hidden_size)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        if output_mean is not None:
            self.output_mean = Parameter(
                torch.from_numpy(output_mean).float()
            )
        else:
            self.output_mean = Parameter(
                torch.rand(self.nb_output_bins.shape).float()
            )

    def forward(self, x):
        # check for waveform or image
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x = self.transform(x)[..., :self.nb_bins]
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # shift and scale input to mean=0 std=1 (across all bins)
        x = x.reshape(nb_frames, nb_samples, nb_channels*nb_bins)

        x = self.in0(x.permute(1, 2, 0)).permute(2, 0, 1)
        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*nb_bins))
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # normalize every instance in a batch
        x = self.in1(x.permute(1, 2, 0)).permute(2, 0, 1)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # reshape to 1D vector (seq_len*batch, hidden_size)
        x = lstm_out[0]

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, self.hidden_size))
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        x = self.in2(x.permute(1, 2, 0)).permute(2, 0, 1)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x.reshape(-1, self.hidden_size))
        x = x.reshape(nb_frames, nb_samples, nb_channels*self.nb_output_bins)
        x = self.in3(x.permute(1, 2, 0)).permute(2, 0, 1)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x)

        return x
