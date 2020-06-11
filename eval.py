import argparse
import musdb
import museval
import test
import multiprocessing
import functools
from model import load_models, Separator
from pathlib import Path
import torch
import tqdm
import utils
import json


def separate_and_evaluate(
    track,
    targets,
    model_name,
    niter,
    output_dir,
    eval_dir,
    residual,
    out,
    device='cpu'
):
    # create the Separator object
    targets = load_models(
        targets=targets,
        model_name=model_name
    )
    separator = Separator(targets=targets,
                          niter=niter,
                          residual=residual,
                          out=out,
                          wiener_win_len=400).to(device)
    separator.freeze()

    audio = utils.preprocess(track.audio, track.rate, separator.sample_rate)

    estimates = separator(audio)
    for key in estimates:
        estimates[key] = estimates[key][0].numpy()
    if output_dir:
        mus.save_estimates(estimates, track, output_dir)

    scores = museval.eval_mus_track(
        track, estimates, output_dir=eval_dir
    )
    print(track, '\n', scores)
    return scores


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='MUSDB18 Evaluation',
        add_help=False
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
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
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
        '--root',
        type=str,
        help='Path to MUSDB18'
    )

    parser.add_argument(
        '--subset',
        type=str,
        default='test',
        help='MUSDB subset (`train`/`test`)'
    )

    parser.add_argument(
        '--cores',
        type=int,
        default=1
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    parser.add_argument(
        '--is-wav',
        action='store_true', default=False,
        help='flags wav version of the dataset'
    )

    args, _ = parser.parse_known_args()
    args = test.inference_args(parser, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mus = musdb.DB(
        root=args.root,
        download=args.root is None,
        subsets=args.subset,
        is_wav=args.is_wav
    )
    out = None if args.out is None else json.loads(args.out)

    if args.cores > 1:
        pool = multiprocessing.Pool(args.cores)
        results = museval.EvalStore()
        scores_list = list(
            pool.imap_unordered(
                func=functools.partial(
                    separate_and_evaluate,
                    targets=args.targets,
                    model_name=args.model,
                    niter=args.niter,
                    residual=args.residual,
                    out=out,
                    output_dir=args.outdir,
                    eval_dir=args.evaldir,
                    device=device
                ),
                iterable=mus.tracks,
                chunksize=1
            )
        )
        pool.close()
        pool.join()
        for scores in scores_list:
            results.add_track(scores)

    else:
        results = museval.EvalStore()
        for track in tqdm.tqdm(mus.tracks):
            scores = separate_and_evaluate(
                track,
                targets=args.targets,
                model_name=args.model,
                niter=args.niter,
                residual=args.residual,
                out=out,
                output_dir=args.outdir,
                eval_dir=args.evaldir,
                device=device
            )
            results.add_track(scores)

    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, args.model)
    method.save(args.model + '.pandas')
