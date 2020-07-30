import pytest
import numpy as np
import torch
import data
import torchaudio
from torchaudio.functional import istft


@pytest.fixture(params=[4096, 4096*10])
def nb_timesteps(request):
    return int(request.param)


@pytest.fixture(params=[1, 2, 3])
def nb_channels(request):
    return request.param


@pytest.fixture
def audio(request, nb_channels, nb_timesteps):
    return torch.rand((nb_channels, nb_timesteps))


def test_gain(audio):
    out = data._augment_gain(audio)
    assert out.shape == audio.shape


def test_channelswap(audio):
    out = data._augment_channelswap(audio)
    assert out.shape == audio.shape


def test_forcestereo(audio):
    out = data._augment_force_stereo(audio)
    assert out.shape == audio.shape
