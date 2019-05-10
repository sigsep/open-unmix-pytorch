import argparse
import musdb
import museval
import test
import multiprocessing
import functools


def separate_and_evaluate(
    track,
    models,
    params,
    niter,
    alpha,
    logit,
    output_dir
):
    print(track.name, track.duration)
    estimates = test.separate_chunked(
        audio=track.audio,
        models=models,
        params=params,
        niter=niter,
        alpha=alpha,
        logit=logit
    )
    if args.outdir:
        mus.save_estimates(estimates, track, args.outdir)

    if args.evaldir is not None:
        scores = museval.eval_mus_track(
            track, estimates, output_dir=args.evaldir
        )
        print(scores)
        return scores


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
        '--root',
        type=str,
        help='Path to MUSDB18'
    )

    parser.add_argument(
        '--cores',
        type=int,
        default=1
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

    models, params = test.load_models(args.model_dir, args.targets)

    mus = musdb.DB(root_dir=args.root, download=False, subsets='test')
    if args.cores > 1:
        pool = multiprocessing.Pool(args.cores)
        results = list(
            pool.imap_unordered(
                func=functools.partial(
                    separate_and_evaluate,
                    models=models,
                    params=params,
                    niter=args.niter,
                    alpha=args.alpha,
                    logit=args.logit,
                    output_dir=args.evaldir
                ),
                iterable=mus.tracks,
                chunksize=1
            )
        )

        pool.close()
        pool.join()
    else:
        for track in mus.tracks:
            separate_and_evaluate(
                track,
                models=models,
                params=params,
                niter=args.niter,
                alpha=args.alpha,
                logit=args.logit,
                output_dir=args.evaldir
            )
