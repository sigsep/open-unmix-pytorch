import pytest
import numpy as np
import os


from umx import data


audio_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data/test.wav',
)


@pytest.fixture(params=["sox", "soundfile", "sox_io"])
def torch_backend(request):
    return request.param


@pytest.fixture(params=[1.0, 2.0, None])
def dur(request):
    return request.param


@pytest.fixture(params=[True, False])
def info(request):
    if request.param:
        return data.load_info(audio_path)
    else:
        return None


def test_loadwav(dur, info):
    audio = data.load_audio(audio_path, dur=dur, info=info)
    rate = 8000
    if dur:
        assert audio.shape[-1] == int(dur * rate)
    else:
        assert audio.shape[-1] == rate * 3
