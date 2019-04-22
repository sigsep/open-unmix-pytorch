import torch
import numpy as np
import argparse
import os.path
import soundfile as sf
import os
import norbert
import json
from pathlib import Path
import scipy.signal

eps = np.finfo(np.float32).eps


def load_models(directory, targets):
    models = {}
    params = {}
    for target_dir in Path(directory).iterdir():
        if target_dir.is_dir():
            if targets is None or target_dir.stem in targets:
                with open(Path(target_dir, 'output.json'), 'r') as stream:
                    results = json.load(stream)

                model = torch.load(
                    Path(target_dir, target_dir.stem + '_model.pth.tar'),
                    map_location='cpu'
                )
                model.to(torch.device("cpu"))
                model.stft.center = True
                # set model into evaluation mode
                model.eval()
                models[target_dir.stem] = model
                params[target_dir.stem] = results
                print("%s model loaded." % target_dir.stem)
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

    # rate = params[list(params.keys())[0]]['rate']
    # seq_dur = params[list(params.keys())[0]]['args']['seq_dur']
    # seq_len = int(seq_dur * rate)

    # split without overlap
    # audio_split = torch.tensor(audio).float().unfold(0, seq_len, seq_len)
    # now its (batch, channels, seq_len/samples)

    # compute STFT of mixture
    # get the first model
    st_model = models[list(models.keys())[0]]

    audio_torch = torch.tensor(audio.T[None, ...]).float()
    # get complex STFT from torch
    X = st_model.stft(audio_torch).detach().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1]*1j
    X = X[0].transpose(2, 1, 0)
    nb_frames_X, nb_bins_X, nb_channels_X = X.shape
    source_names = []
    V = []
    for j, (target, model) in enumerate(models.items()):
        Vj = model(
            torch.tensor(audio.T[None, ...]).float()
        ).cpu().detach().numpy()**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

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
        estimates[name] = audio_hat.T

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
