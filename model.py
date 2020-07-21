from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import json
import model
from filtering import wiener
from torchaudio.functional import istft
import torchaudio
from typing import Optional


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
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

        shape = x.size()
        nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])

        stft_f = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            pad_mode='reflect'
        )

        # unpack batch
        stft_f = stft_f.view(shape[:-1] + stft_f.shape[-3:])
        # bring frames to first dimension for efficient lstm
        return stft_f


class ComplexNorm(nn.Module):
    def __init__(
        self,
        power=1,
        mono=False
    ):
        super(ComplexNorm, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_samples, nb_channels, nb_bins, nb_frames)
        """
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain to preserve energy
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        return stft_f


class OpenUnmix(nn.Module):
    def __init__(
        self,
        nb_bins=4096,
        nb_channels=2,
        hidden_size=512,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        unidirectional=False,
        max_bin=None
    ):
        """
        Input:  (nb_samples, nb_channels, nb_bins, nb_frames)
        Output: (nb_samples, nb_channels, nb_bins, nb_frames)
        """

        super(OpenUnmix, self).__init__()

        self.nb_output_bins = nb_bins
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4 if nb_layers > 1 else 0,
        )

        fc2_hiddensize = hidden_size * 2
        self.fc2 = Linear(
            in_features=fc2_hiddensize,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins*nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        # permute so that batch is last for lstm
        x = x.permute(3, 0, 1, 2)
        # get current spectrogram shape
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., :self.nb_bins]
        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return x.permute(1, 2, 3, 0)


class Separator(nn.Module):
    """
    Separator class to encapsulate all the stereo filtering
    as a torch Module, to enable end-to-end learning.

    Parameters
    ----------
    targets: dictionary of target models {target: model}
        the spectrogram models to be used by the Separator. Each model
        may for instance be loaded with the `utils.load_target_models` function

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage. Zeroed if only one target is estimated.
         defaults to 1.

    residual: bool
        adds an additional residual target, obtained by
        subtracting the other estimated targets from the mixture, before any
        potential EM post-processing.
        Defaults to False

    wiener_win_len: {None | int}
        The size of the excerpts (number of frames) on which to apply filtering
        independently. This means assuming time varying stereo models and
        localization of sources.
        None means not batching but using the whole signal. It comes at the
        price of a much larger memory usage.
    """
    def __init__(
        self,
        target_models: dict,
        niter: int = 0,
        softmask: bool = False,
        residual: bool = False,
        sample_rate: int = 44100,
        n_fft: int = 4096,
        n_hop: int = 1024,
        nb_channels: int = 2,
        wiener_win_len: Optional[int] = 300
    ):
        super(Separator, self).__init__()

        # saving parameters
        self.niter = niter
        self.residual = residual
        self.softmask = softmask
        self.wiener_win_len = wiener_win_len

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop, center=True)
        self.complexnorm = ComplexNorm(power=1, mono=nb_channels == 1)

        # registering the targets models
        self.target_models = nn.ModuleDict(target_models)
        # adding till https://github.com/pytorch/pytorch/issues/38963
        self.nb_targets = len(self.target_models)
        # get the sample_rate as the sample_rate of the first model
        # (tacitly assume it's the same for all targets)
        self.register_buffer('sample_rate', torch.as_tensor(sample_rate))

    def freeze(self):
        # set all parameters as not requiring gradient, more RAM-efficient
        # at test time
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, audio):
        """Performing the separation on audio input

        Parameters
        ----------
        audio: torch.Tensor [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio

        Returns
        -------
        estimates: `torch.Tensor`
                   shape(nb_samples, nb_targets, nb_channels, nb_timesteps)
        """

        # initializing spectrograms variable
        spectrograms = None

        nb_sources = self.nb_targets
        nb_samples = audio.shape[0]

        # getting the STFT of mix:
        # (nb_samples, nb_channels, nb_bins, nb_frames, 2)
        mix_stft = self.stft(audio)
        X = self.complexnorm(mix_stft)
        for j, (target_name, target_module) in enumerate(
            self.target_models.items()
        ):

            # apply current model to get the source spectrogram
            target_spectrogram = target_module(X.detach().clone())

            # output is nb_frames, nb_samples, nb_channels, nb_bins
            if spectrograms is None:
                # allocate the spectrograms variable
                spectrograms = torch.zeros(
                    target_spectrogram.shape + (nb_sources,),
                    dtype=audio.dtype,
                    device=target_spectrogram.device
                )

            spectrograms[..., j] = target_spectrogram

        # transposing it as
        # (nb_samples, nb_frames, nb_bins,{1,nb_channels}, nb_sources)
        spectrograms = spectrograms.permute(0, 3, 2, 1, 4)

        # rearranging it into:
        # (nb_samples, nb_frames, nb_bins, nb_channels, 2) to feed
        # into filtering methods
        mix_stft = mix_stft.permute(0, 3, 2, 1, 4)

        # create an additional target if we need to build a residual
        if self.residual:
            # we add an additional target
            nb_sources += 1

        if nb_sources == 1 and self.niter > 0:
            raise Exception('Cannot use EM if only one target is estimated.'
                            'Provide two targets or create an additional '
                            'one with `--residual`')

        nb_frames = spectrograms.shape[1]
        targets_stft = torch.zeros(
            mix_stft.shape + (nb_sources,),
            dtype=audio.dtype,
            device=mix_stft.device
        )
        for sample in range(nb_samples):
            pos = 0
            if self.wiener_win_len:
                wiener_win_len = self.wiener_win_len
            else:
                wiener_win_len = nb_frames
            while pos < nb_frames:
                t = torch.arange(pos, min(nb_frames, pos+wiener_win_len))
                pos = t[-1] + 1

                targets_stft[sample, t] = wiener(
                    spectrograms[sample, t],
                    mix_stft[sample, t],
                    self.niter,
                    softmask=self.softmask,
                    residual=self.residual
                )

        # getting to (nb_samples, nb_targets, channel, fft_size, n_frames, 2)
        targets_stft = targets_stft.permute(0, 5, 3, 2, 1, 4).contiguous()

        # inverse STFTs
        estimates = istft(
            targets_stft,
            n_fft=self.stft.n_fft,
            hop_length=self.stft.n_hop,
            window=self.stft.window,
            center=self.stft.center,
            normalized=False,
            onesided=True,
            pad_mode='reflect',
            length=audio.shape[-1]
        )

        # returning (nb_samples, nb_targets, nb_channels, nb_timesteps)
        return estimates

    def to_dict(
        self,
        estimates: dict,
        aggregate_dict: Optional[dict] = None
    ) -> dict:
        """Convert estimates as stacked tensor to dictionary

        Parameters
        ----------
        estimates:      Dict
        aggregate_dict: Dict

        Returns
        -------
        estimates_dict: Dict
        """
        estimates_dict = {}
        for k, target in enumerate(self.target_models):
            estimates_dict[target] = estimates[:, k, ...]

        # in the case of residual, we added another source
        if self.residual:
            estimates_dict['residual'] = estimates[:, -1, ...]

        if aggregate_dict is not None:
            new_estimates = {}
            for key in aggregate_dict:
                new_estimates[key] = 0.0
                for target in aggregate_dict[key]:
                    new_estimates[key] = new_estimates[key] + \
                        estimates_dict[target]
            estimates_dict = new_estimates
        return estimates_dict
