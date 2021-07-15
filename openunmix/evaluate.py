import argparse
import functools
import json
import multiprocessing
from typing import Optional, Union

import musdb
import museval
import torch
import tqdm

from openunmix import utils


def separate_and_evaluate(
    track: musdb.MultiTrack,
    targets: list,
    model_str_or_path: str,
    niter: int,
    output_dir: str,
    eval_dir: str,
    residual: bool,
    mus,
    aggregate_dict: dict = None,
    device: Union[str, torch.device] = "cpu",
    wiener_win_len: Optional[int] = None,
    filterbank="torch",
) -> str:

    separator = utils.load_separator(
        model_str_or_path=model_str_or_path,
        targets=targets,
        niter=niter,
        residual=residual,
        wiener_win_len=wiener_win_len,
        device=device,
        pretrained=True,
        filterbank=filterbank,
    )

    separator.freeze()
    separator.to(device)

    audio = torch.as_tensor(track.audio, dtype=torch.float32, device=device)
    audio = utils.preprocess(audio, track.rate, separator.sample_rate)

    estimates = separator(audio)
    estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)

    for key in estimates:
        estimates[key] = estimates[key][0].cpu().detach().numpy().T
    if output_dir:
        mus.save_estimates(estimates, track, output_dir)

    scores = museval.eval_mus_track(track, estimates, output_dir=eval_dir)
    return scores


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="MUSDB18 Evaluation", add_help=False)

    parser.add_argument(
        "--targets",
        nargs="+",
        default=["vocals", "drums", "bass", "other"],
        type=str,
        help="provide targets to be processed. \
              If none, all available targets will be computed",
    )

    parser.add_argument(
        "--model",
        default="umxl",
        type=str,
        help="path to mode base directory of pretrained models",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        help="Results path where audio evaluation results are stored",
    )

    parser.add_argument("--evaldir", type=str, help="Results path for museval estimates")

    parser.add_argument("--root", type=str, help="Path to MUSDB18")

    parser.add_argument("--subset", type=str, default="test", help="MUSDB subset (`train`/`test`)")

    parser.add_argument("--cores", type=int, default=1)

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA inference"
    )

    parser.add_argument(
        "--is-wav",
        action="store_true",
        default=False,
        help="flags wav version of the dataset",
    )

    parser.add_argument(
        "--niter",
        type=int,
        default=1,
        help="number of iterations for refining results.",
    )

    parser.add_argument(
        "--wiener-win-len",
        type=int,
        default=300,
        help="Number of frames on which to apply filtering independently",
    )

    parser.add_argument(
        "--residual",
        type=str,
        default=None,
        help="if provided, build a source with given name"
        "for the mix minus all estimated targets",
    )

    parser.add_argument(
        "--aggregate",
        type=str,
        default=None,
        help="if provided, must be a string containing a valid expression for "
        "a dictionary, with keys as output target names, and values "
        "a list of targets that are used to build it. For instance: "
        '\'{"vocals":["vocals"], "accompaniment":["drums",'
        '"bass","other"]}\'',
    )

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    mus = musdb.DB(
        root=args.root,
        download=args.root is None,
        subsets=args.subset,
        is_wav=args.is_wav,
    )
    aggregate_dict = None if args.aggregate is None else json.loads(args.aggregate)

    if args.cores > 1:
        pool = multiprocessing.Pool(args.cores)
        results = museval.EvalStore()
        scores_list = list(
            pool.imap_unordered(
                func=functools.partial(
                    separate_and_evaluate,
                    targets=args.targets,
                    model_str_or_path=args.model,
                    niter=args.niter,
                    residual=args.residual,
                    mus=mus,
                    aggregate_dict=aggregate_dict,
                    output_dir=args.outdir,
                    eval_dir=args.evaldir,
                    device=device,
                ),
                iterable=mus.tracks,
                chunksize=1,
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
                model_str_or_path=args.model,
                niter=args.niter,
                residual=args.residual,
                mus=mus,
                aggregate_dict=aggregate_dict,
                output_dir=args.outdir,
                eval_dir=args.evaldir,
                device=device,
            )
            print(track, "\n", scores)
            results.add_track(scores)

    print(results)
    method = museval.MethodStore()
    method.add_evalstore(results, args.model)
    method.save(args.model + ".pandas")
