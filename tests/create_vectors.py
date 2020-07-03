import os
import torch
import musdb
import numpy as np
import utils
import model

"""script to create spectrogram test vectors for STFT regression tests"""


def main():
    test_track = 'Al James - Schoolboy Facination'
    mus = musdb.DB(download=True)

    # load audio track
    track = [track for track in mus.tracks if track.name == test_track][0]

    # convert to torch tensor
    audio = utils.preprocess(track.audio, track.rate, track.rate)

    stft = model.STFT(n_fft=4096, n_hop=1024)
    spec = model.Spectrogram(power=1, mono=False)
    magnitude_spectrogram = spec(stft(audio))
    torch.save(
        magnitude_spectrogram,
        'Al James - Schoolboy Facination.spectrogram.pt'
    )


if __name__ == "__main__":
    main()
