import os
import pytest
import musdb
import simplejson as json
import numpy as np
import torch


from openunmix import model
from openunmix import evaluate
from openunmix import utils
from openunmix import transforms


test_track = "Al James - Schoolboy Facination"

json_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data/%s.json" % test_track,
)

spec_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data/%s.spectrogram.pt" % test_track,
)


@pytest.fixture(params=["torch", "asteroid"])
def method(request):
    return request.param


@pytest.fixture()
def mus():
    return musdb.DB(download=True)


def test_estimate_and_evaluate(mus):
    # return any number of targets
    with open(json_path) as json_file:
        ref = json.loads(json_file.read())

    track = [track for track in mus.tracks if track.name == test_track][0]

    scores = evaluate.separate_and_evaluate(
        track,
        targets=["vocals", "drums", "bass", "other"],
        model_str_or_path="umx",
        niter=1,
        residual=None,
        mus=mus,
        aggregate_dict=None,
        output_dir=None,
        eval_dir=None,
        device="cpu",
        wiener_win_len=None,
    )

    assert scores.validate() is None

    with open(os.path.join(".", track.name) + ".json", "w+") as f:
        f.write(scores.json)

    scores = json.loads(scores.json)

    for target in ref["targets"]:
        for metric in ["SDR", "SIR", "SAR", "ISR"]:

            ref = np.array([d["metrics"][metric] for d in target["frames"]])
            idx = [t["name"] for t in scores["targets"]].index(target["name"])
            est = np.array([d["metrics"][metric] for d in scores["targets"][idx]["frames"]])

            assert np.allclose(ref, est, atol=1e-01)


def test_spectrogram(mus, method):
    """Regression test for spectrogram transform

    Loads pre-computed transform and compare to current spectrogram
    e.g. this makes sure that the training is reproducible if parameters
    such as STFT centering would be subject to change.
    """
    track = [track for track in mus.tracks if track.name == test_track][0]

    stft, _ = transforms.make_filterbanks(
        n_fft=4096, n_hop=1024, sample_rate=track.rate, method=method
    )
    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=False))
    audio = torch.as_tensor(track.audio, dtype=torch.float32, device="cpu")
    audio = utils.preprocess(audio, track.rate, track.rate)
    ref = torch.load(spec_path)
    dut = encoder(audio).permute(3, 0, 1, 2)

    assert torch.allclose(ref, dut, atol=1e-4, rtol=1e-3)
