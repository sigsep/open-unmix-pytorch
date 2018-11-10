import norbert
import musdb
import tqdm
import os
import argparse
import numpy as np
tqdm.monitor_interval = 0

targets_to_process = ['vocals', 'drums', 'other', 'bass', 'accompaniment']

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MUSMAG')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='nfft')
    parser.add_argument('--hop', type=int, default=1024,
                        help='hop size')
    parser.add_argument('--root_dir', type=str, default=".",
                        help='provide output path base folder name')
    parser.add_argument('--iswav', help='Read musdb wav instead of stems',
                        action='store_true')
    args = parser.parse_args()

    estimates_dir = 'musmag-npy'

    mus = musdb.DB(is_wav=args.iswav)
    tracks = mus.load_mus_tracks()
    for track in tqdm.tqdm(tracks):
        # set (trackwise) norbert objects
        tf = norbert.TF(n_fft=args.nfft, n_hopsize=args.hop)

        def pipeline(t, mono=True, bounds=None):
            x = np.abs(tf.transform(t.audio))
            if mono:
                x = np.sqrt(np.sum(np.abs(x)**2, axis=-1, keepdims=True))

            return x.astype(np.float32)

        X = pipeline(track)

        track_estimate_dir = os.path.join(
            args.root_dir, estimates_dir, track.subset, track.name
        )
        if not os.path.exists(track_estimate_dir):
            os.makedirs(track_estimate_dir)

        # write out tracks to disk

        np.save(os.path.join(track_estimate_dir, 'mix.npy'), X)
        for name, track in track.targets.items():
            if name in targets_to_process:
                S = pipeline(track)
                np.save(os.path.join(track_estimate_dir, name + '.npy'), S)
