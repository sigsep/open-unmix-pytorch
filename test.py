from __future__ import print_function
import norbert
import torch
import numpy as np
import argparse
import musdb
import functools
import museval
import os.path
import soundfile as sf
import os
from norbert.contrib import split, overlapadd
import networks
import json
from pathlib import Path


eps = np.finfo(np.float32).eps


def load_models(directory, targets):
    models = {}
    params = {}
    for target_dir in Path(directory).iterdir():
        if target_dir.is_dir():
            if targets is None or target_dir.stem in targets:
                with open(Path(target_dir, 'output.json'), 'r') as stream:
                    results = json.load(stream)

                state = torch.load(
                    Path(target_dir, target_dir.stem + '.pth.tar'),
                    map_location='cpu'
                )['state_dict']

                network = networks.load(results['args']['network'])

                try:
                    input_mean = state['input_mean']
                except KeyError:
                    input_mean = None

                try:
                    input_scale = state['input_scale']
                except KeyError:
                    input_scale = None

                try:
                    output_mean = state['output_mean']
                except KeyError:
                    output_mean = None

                try:
                    dropout = results['args']['dropout']
                except KeyError:
                    dropout = None

                model = network.Model(
                    results['nb_features'],
                    results['args']['ex_len'],
                    output_mean=output_mean,
                    input_mean=input_mean,
                    input_scale=input_scale,
                    nb_layers=results['args']['nb_lstm_layers'],
                    hidden_size=results['args']['hidden_size'],
                    uni=results['args']['uni'],
                    dropout=dropout
                ).to(torch.device("cpu"))
                model.load_state_dict(state)

                # set model into evaluation mode
                model.eval()
                models[target_dir.stem] = model
                params[target_dir.stem] = results
    return models, params


def separate(
    audio, models, params,
    niter=0, alpha=2,
    mono=False,
    logit=0,
    smoothing=0,
    nfft=4096, hop=1024
):
    tf = norbert.TF(n_fft=nfft, n_hopsize=hop)

    X = tf.transform(audio)

    (nb_frames, nb_bins, nb_channels) = X.shape

    Xm = np.abs(X).astype(np.float32)
    if mono:
        Xm = np.sqrt(np.sum(Xm**2, axis=-1, keepdims=True))

    Xm_shape = Xm.shape
    # for now only check the first model, as they are assumed to be the same
    ex_len = params[list(params.keys())[0]]['args']['ex_len']
    # split without overlap
    Xm = split(Xm, ex_len, ex_len, weight_frames=False)
    #  getting to nb_batches, ex_len, nb_features, nb_channels
    Xm = np.transpose(Xm, (0, 1, 3, 2))
    nb_sources = len(models)
    source_names = []
    V = np.zeros((nb_frames, nb_bins, nb_channels, nb_sources), np.float32)
    for j, (target, model) in enumerate(models.items()):
        Vj = model(
            torch.tensor(Xm, dtype=torch.float32)
        ).cpu().detach().numpy()**alpha
        #  getting to ex_len, nb_batches, nb_features, nb_channels
        Vj = np.transpose(Vj, (0, 1, 3, 2))
        Vj = overlapadd(Vj, 1, ex_len, Xm_shape, weighted_frames=False)
        V[..., j] = Vj
        source_names += [target]

    if nb_sources == 1:
        V = norbert.add_residual_model(V, X, alpha)
        source_names += ['accompaniment']

    if not logit:
        logit = None
        Y = norbert.wiener(np.copy(V), np.copy(X), niter,
                       smoothing=smoothing, logit=logit)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = tf.inverse_transform(Y[..., j])
        estimates[name] = audio_hat

    return estimates


def musdb_separate(
    track, models, params, niter, alpha, mono, logit,
    smoothing, nfft, hop, eval_dir=None
):
    estimates = separate(
        track.audio, models, params,
        niter=niter, alpha=alpha, mono=mono, logit=logit, smoothing=smoothing, nfft=nfft, hop=hop,
    )
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
        '--nfft',
        type=int,
        default=4096,
        help='nfft size'
    )

    parser.add_argument(
        '--hop',
        type=int,
        default=1024,
        help='stft hop size'
    )

    parser.add_argument(
        '--alpha',
        type=int,
        default=1,
        help='exponent for softmasks'
    )

    parser.add_argument(
        '--mono',
        action='store_true',
        default=False,
        help='process mono'
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
        mus = musdb.DB()
        mus.run(
            functools.partial(
                musdb_separate,
                models=models,
                params=params,
                niter=args.niter,
                alpha=args.alpha,
                mono=args.mono,
                logit=args.logit,
                smoothing=args.smoothing,
                nfft=args.nfft,
                hop=args.hop,
                eval_dir=args.evaldir,
            ),
            estimates_dir=args.outdir,
            subsets='test',
            parallel=False,
            cpus=4
        )
    else:
        # handling an input wav file
        audio, samplerate = sf.read(args.input)
        estimates = separate(
            audio,
            models,
            params,
            niter=args.niter,
            alpha=args.alpha,
            mono=args.mono,
            logit=args.logit,
            smoothing=args.smoothing,
            nfft=args.nfft,
            hop=args.hop
        )
        base = os.path.basename(args.input)

        for key in estimates:
            sf.write(key+'_' + base, estimates[key], samplerate)
