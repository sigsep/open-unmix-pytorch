import torch
import numpy as np
import argparse
import os.path
import soundfile as sf
import os
import norbert
import json
import model
from pathlib import Path
import scipy.signal
import torch.nn.functional as F

eps = np.finfo(np.float32).eps


def load_models(directory, targets):
    models = {}
    params = {}
    for target_dir in Path(directory).iterdir():
        if target_dir.is_dir():
            if targets is None or target_dir.stem in targets:
                with open(Path(target_dir, 'output.json'), 'r') as stream:
                    results = json.load(stream)

                unmix = torch.load(
                    Path(target_dir, target_dir.stem + '_model.pth.tar'),
                    map_location='cpu'
                )
                unmix.to(torch.device("cpu"))
                # set model into evaluation mode
                unmix.eval()
                models[target_dir.stem] = unmix
                params[target_dir.stem] = results
    return models, params


def istft(X, rate=44100, n_fft=2048, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(audio, models, params, niter=0, alpha=2, logit=0):
    # for now only check the first model, as they are assumed to be the same
    nb_sources = len(models)
    # get the first model
    st_model = models[list(models.keys())[0]]

    rate = params[list(params.keys())[0]]['rate']
    seq_dur = params[list(params.keys())[0]]['args']['seq_dur']
    seq_len = int(seq_dur * rate)
    # correct sequence length to multiple of n_fft
    seq_len -= seq_len % st_model.stft.n_fft

    # split without overlap
    # now its (batch, channels, seq_len/samples)

    # compute STFT of mixture
    audio = torch.tensor(audio.T).float()
    audio_shape = audio.shape
    paddings = (0, seq_len - (audio_shape[-1] % seq_len))
    audio_padded = F.pad(audio, paddings, "constant", 0)
    audio_split = audio_padded.unfold(1, seq_len, seq_len)
    audio_split = audio_split.permute(1, 0, 2)

    # audio_torch = torch.tensor(audio.T[None, ...]).float()
    # get complex STFT from torch
    X = st_model.stft(audio_split)
    # precompute mixture spectrogram
    M = st_model.spec(X)

    X = X.detach().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1]*1j
    X = X.transpose(0, 3, 2, 1)

    # Run unmix
    source_names = []
    V = []
    for j, (target, unmix) in enumerate(models.items()):
        unmix.transform = model.NoOp()
        Vj = unmix(M.clone()).cpu().detach().numpy()**alpha

        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj)  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (2, 1, 4, 3, 0))
    V = V.reshape(-1, *V.shape[2:])
    X = X.reshape(-1, *X.shape[2:])
    if nb_sources == 1:
        V = norbert.residual(V, X, alpha)
        source_names += ['accompaniment']

    if not logit:
        logit = None
        Y = norbert.wiener(V, X, niter, logit=logit)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=st_model.stft.n_fft,
            n_hopsize=st_model.stft.n_hop
        )
        estimates[name] = audio_hat[:, :-paddings[-1]].T

    return estimates


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='OSU Inference')

    parser.add_argument(
        'model_dir',
        type=str,
        default=".",
        help='path to models'
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        type=str,
        help='provide targets to be processed. \
              If none, all available targets will be computed'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        default="OSU_RESULTS",
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--evaldir',
        type=str,
        help='Results path for museval estimates'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Path to wav file. If not provided, will process the MUSDB18'
    )

    parser.add_argument(
        '--niter',
        type=int,
        default=0,
        help='number of iterations. 0 is softmask'
    )

    parser.add_argument(
        '--alpha',
        type=int,
        default=1,
        help='exponent for softmasks'
    )

    parser.add_argument(
        '--logit',
        type=float,
        help='apply logit compression. 0 means no compression'
    )

    args = parser.parse_args()

    models, params = load_models(args.model_dir, args.targets)

    # handling an input audio path
    audio, samplerate = sf.read(args.input)
    estimates = separate(
        audio,
        models,
        params,
        niter=args.niter,
        alpha=args.alpha,
        logit=args.logit
    )
    base = os.path.basename(args.input)

    for key in estimates:
        sf.write(key+'_' + base, estimates[key], samplerate)
