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
import warnings


def load_models(directory, targets, device='cpu'):
    models = {}
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
                print("%s model loaded." % target_dir.stem)
    return models


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(audio, models, niter=0, softmask=False, alpha=1,
             residual_model=False, device='cpu'):
    # for now only check the first model, as they are assumed to be the same
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

    if residual_model or len(models) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += (['residual'] if len(models) > 1
                         else ['accompaniment'])

    Y = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=st_model.stft.n_fft,
            n_hopsize=st_model.stft.n_hop
        )
        estimates[name] = audio_hat.T

    return estimates


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    inf_parser.add_argument(
        '--softmask',
        dest='softmask',
        action='store_true',
        help=('if enabled, will initialize separation with softmask.'
              'otherwise, will use mixture phase with spectrogram')
    )

    inf_parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--alpha',
        type=int,
        default=1,
        help='exponent in case of softmask separation'
    )

    inf_parser.add_argument(
        '--samplerate',
        type=int,
        default=44100,
        help='model samplerate'
    )

    inf_parser.add_argument(
        '--residual-model',
        action='store_true',
        help='create a model for the residual'
    )
    return inf_parser.parse_args()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='OSU Inference',
        add_help=False
    )

    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='List of paths to wav/flac files.'
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        default=['vocals', 'drums', 'bass', 'other'],
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
        '--modeldir',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--modelname',
        choices=['OpenUnmixStereo'],
        default='OpenUnmixStereo',
        type=str,
        help='use pretrained model'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.modeldir:
        models = load_models(args.modeldir, args.targets, device=device)
        model_name = Path(args.modeldir).stem
    else:
        import hubconf
        pretrained_model = getattr(hubconf, args.modelname)
        models = {
            target: pretrained_model(target=target, device=device)
            for target in args.targets
        }
        model_name = args.modelname

    for input_file in args.input:
        if not args.outdir:
            outdir = Path(Path(input_file).stem + '_' + model_name)
        else:
            outdir = Path(args.outdir)

        # handling an input audio path
        audio, rate = sf.read(input_file, always_2d=True)

        if audio.shape[1] > 2:
            warnings.warn('Channel count > 2! '
            'Only the first two channels will be processed!')
            audio = audio[:, :2]

        if rate != args.samplerate:
            # resample to model samplerate if needed
            audio = resampy.resample(audio, rate, args.samplerate, axis=0)

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            audio = np.repeat(audio, 2, axis=1)

        estimates = separate(
            audio,
            models,
            niter=args.niter,
            alpha=args.alpha,
            softmask=args.softmask,
            residual_model=args.residual_model,
            device=device
        )
        outdir.mkdir(exist_ok=True, parents=True)
        for target in estimates:
            sf.write(
                outdir / Path(target).with_suffix('.wav'),
                estimates[target],
                args.samplerate
            )
