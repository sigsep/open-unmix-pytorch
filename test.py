import torch
import numpy as np
import argparse
import soundfile as sf
import norbert
import json
from pathlib import Path
import scipy.signal
import resampy
import model
import utils

eps = np.finfo(np.float32).eps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                    map_location=device
                )['state_dict']

                max_bin = utils.bandwidth_to_max_bin(
                    state['sample_rate'],
                    results['args']['nfft'],
                    results['args']['bandwidth']
                )

                unmix = model.OpenUnmix(
                    n_fft=results['args']['nfft'],
                    n_hop=results['args']['nhop'],
                    nb_channels=results['args']['nb_channels'],
                    hidden_size=results['args']['hidden_size'],
                    max_bin=max_bin
                )

                unmix.load_state_dict(state)
                unmix.stft.center = True
                # set model into evaluation mode
                unmix.eval()
                unmix.to(device)
                models[target_dir.stem] = unmix
                params[target_dir.stem] = results
                print("%s model loaded." % target_dir.stem)
    return models, params


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(audio, models, params, niter=0, softmask=0, alpha=1,
             final_smoothing=0):
    # for now only check the first model, as they are assumed to be the same
    nb_sources = len(models)
    st_model = models[list(models.keys())[0]]

    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)
    # get complex STFT from torch
    X = st_model.stft(audio_torch).detach().cpu().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1]*1j
    X = X[0].transpose(2, 1, 0)
    nb_frames_X, nb_bins_X, nb_channels_X = X.shape
    source_names = []
    V = []
    for j, (target, unmix) in enumerate(models.items()):
        Vj = unmix(audio_torch).cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    if nb_sources == 1:
        V = norbert.residual(V, X, alpha)
        source_names += ['accompaniment']

    Y = norbert.wiener(V, X, niter, softmask=softmask,
                       final_smoothing=final_smoothing)

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
        'modeldir',
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
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--evaldir',
        type=str,
        help='Results path for museval estimates'
    )

    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='List of paths to wav files. '
             'If not provided, will process the MUSDB18'
    )

    parser.add_argument(
        '--softmask',
        dest='softmask',
        action='store_true',
        help=('will use mixture phase with spectrogram'
              'estimates, if enabled')
    )

    parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    parser.add_argument(
        '--alpha',
        type=int,
        default=1,
        help='exponent in case of softmask separation'
    )

    parser.add_argument(
        '--samplerate',
        type=int,
        default=44100,
        help='model samplerate'
    )

    parser.add_argument(
        '--final-smoothing',
        type=int,
        default=0,
        help=('final smoothing of estimates. Reduces distortion, adds '
              'interference')
    )

    args = parser.parse_args()

    models, params = load_models(args.modeldir, args.targets)

    for input_file in args.input:
        if not args.outdir:
            outdir = Path(
                Path(input_file).stem + '_' + Path(args.modeldir).stem
            )
        else:
            outdir = Path(args.outdir)

        print('Processing ', input_file)
        # handling an input audio path
        audio, rate = sf.read(input_file, always_2d=True)
        # todo: implement other sample rates

        if rate != args.samplerate:
            audio = resampy.resample(audio, rate, args.samplerate, axis=0)

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            audio = np.repeat(audio, 2, axis=1)

        estimates = separate(
            audio,
            models,
            params,
            niter=args.niter,
            alpha=args.alpha,
            softmask=args.softmask,
            final_smoothing=args.final_smoothing
        )
        outdir.mkdir(exist_ok=True, parents=True)
        for target in estimates:
            sf.write(
                outdir / Path(target).with_suffix('.wav'),
                estimates[target],
                args.samplerate
            )
