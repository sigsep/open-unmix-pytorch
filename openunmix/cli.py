from pathlib import Path
import torch
import torchaudio
import json

from openunmix import utils
from openunmix import predict

import argparse


def separate():
    parser = argparse.ArgumentParser(
        description='UMX Inference',
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        '--start',
        type=float,
        default=0.0,
        help='Audio chunk start in seconds'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=-1.0,
        help='Audio chunk duration in seconds, negative values load full track'
    )

    parser.add_argument(
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    parser.add_argument(
        '--audio-backend',
        type=str,
        default="soundfile",
        help='Set torchaudio backend '
             '(`sox_io`, `sox` or `soundfile`), defaults to `soundfile`')

    parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    parser.add_argument(
        '--wiener-win-len',
        type=int,
        default=300,
        help='Number of frames on which to apply filtering independently'
    )

    parser.add_argument(
        '--residual',
        type=str,
        default=None,
        help='if provided, build a source with given name'
             'for the mix minus all estimated targets'
    )

    parser.add_argument(
        '--aggregate',
        type=str,
        default=None,
        help='if provided, must be a string containing a valid expression for '
             'a dictionary, with keys as output target names, and values '
             'a list of targets that are used to build it. For instance: '
             '\'{\"vocals\":[\"vocals\"], \"accompaniment\":[\"drums\",'
             '\"bass\",\"other\"]}\''
    )

    args = parser.parse_args()

    torchaudio.set_audio_backend(args.audio_backend)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # parsing the output dict
    aggregate_dict = None if args.aggregate is None else json.loads(
        args.aggregate
    )

    # create separator only once to reduce model loading
    # when using multiple files
    separator = utils.load_separator(
        model_str_or_path=args.model,
        targets=args.targets,
        niter=args.niter,
        residual=args.residual,
        wiener_win_len=args.wiener_win_len,
        device=device,
        pretrained=True
    )

    separator.freeze()
    separator.to(device)

    # loop over the files
    for input_file in args.input:
        estimates = predict.separate(
            input_file,
            aggregate_dict=aggregate_dict,
            separator=separator,
            device=device
        )
        if not args.outdir:
            model_path = Path(args.model)
            if not model_path.exists():
                outdir = Path(Path(input_file).stem + '_' + args.model)
            else:
                outdir = Path(Path(input_file).stem + '_' + model_path.stem)
        else:
            outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        # write out estimates
        for target, estimate in estimates.items():
            target_path = str(outdir / Path(target).with_suffix('.wav'))
            torchaudio.save(
                 target_path,
                 torch.squeeze(estimate).to('cpu'),
                 separator.sample_rate
            )
