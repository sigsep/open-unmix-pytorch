import torch
import numpy as np
import argparse
import musdb
import museval
import os.path
import soundfile as sf
import os
from model import OSU
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
                # set model into evaluation mode
                model.eval()
                models[target_dir.stem] = model
                params[target_dir.stem] = results
    return models, params


def istft(X, rate=44100, n_fft=2048, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / n_hopsize,
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(
    audio, models, params,
    niter=0, alpha=2,
    mono=False,
    logit=0,
    smoothing=0
):
    # for now only check the first model, as they are assumed to be the same
    nb_sources = len(models)

    rate = params[list(params.keys())[0]]['rate']
    seq_dur = params[list(params.keys())[0]]['args']['seq_dur']
    seq_len = int(seq_dur * rate)

    # split without overlap
    # audio_split = torch.tensor(audio).float().unfold(0, seq_len, seq_len)
    # now its (batch, channels, seq_len/samples)

    # compute STFT of mixture
    # get the first model
    st_model = models[list(models.keys())[0]]

    # get complex STFT from torch
    X = st_model.transform[0](torch.tensor(audio.T[None, ...]).float())
    # convert to complex numpy type
    X = X.cpu().detach().numpy()[..., 0] + X.cpu().detach().numpy()[..., 1]*1j
    X = X[0].transpose(2, 1, 0)
    nb_frames, nb_bins, nb_channels = X.shape
    source_names = []
    V = np.zeros((nb_frames, nb_bins, nb_sources), np.float32)
    for j, (target, model) in enumerate(models.items()):
        Vj = model(
            torch.tensor(audio.T[None, ...]).float()
        ).cpu().detach().numpy()**alpha

        #  transposing to ex_len, nb_batches, nb_features, nb_channels
        Vj = np.transpose(Vj, (0, 3, 1, 2))
        # TODO: fold signal
        V[..., j] = Vj[..., 0, 0]
        source_names += [target]

    if nb_sources == 1:
        V = norbert.residual(V[..., None, :], X, alpha)
        source_names += ['accompaniment']

    if not logit:
        logit = None
        Y = norbert.wiener(
            np.copy(V), np.copy(X), niter,
            smoothing=smoothing, logit=logit
        )

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(Y[..., j].T)
        estimates[name] = audio_hat.T

    return estimates


def musdb_separate(
    track, models, params, eval_dir=None, *args, **kwargs
):
    estimates = separate(track.audio, models, params, *args, **kwargs)
    # Evaluate using museval
    if eval_dir is not None:
        scores = museval.eval_mus_track(
            track, estimates, output_dir=eval_dir
        )
        print(scores)

    return estimates


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Inference Example')

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
        help='output path for estimates'
    )

    parser.add_argument(
        '--evaldir',
        type=str,
        help='output path for estimates'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='link to wav file. If not provided, will process the musdb'
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

    parser.add_argument(
        '--smoothing',
        type=int,
        help='apply smoothing during EM. 0 means no smoothing'
    )

    args = parser.parse_args()

    models, params = load_models(args.model_dir, args.targets)

    if args.input is None:
        # handling the MUSDB case
        mus = musdb.DB(download=True)
        for track in mus:
            estimates = musdb_separate(
                track=track,
                models=models,
                params=params,
                niter=args.niter,
                alpha=args.alpha,
                logit=args.logit,
                smoothing=args.smoothing,
                eval_dir=args.evaldir
            )
            mus.save_estimates(estimates, track, 'OSU alpha1')
            print(track)
    else:
        # handling an input wav file
        audio, samplerate = sf.read(args.input)
        estimates = separate(
            audio,
            models,
            params,
            niter=args.niter,
            alpha=args.alpha,
            logit=args.logit,
            smoothing=args.smoothing,
        )
        base = os.path.basename(args.input)

        for key in estimates:
            sf.write(key+'_' + base, estimates[key], samplerate)
