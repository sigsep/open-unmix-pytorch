import os
import torch
import musdb
import numpy as np
from openunmix import model, utils

"""script to create spectrogram test vectors for STFT regression tests

Test vectors have been created using the `v1.0.0` release tag as this
was the commit that umx was trained with
"""


def main():
    test_track = "Al James - Schoolboy Facination"
    mus = musdb.DB(download=True)

    # load audio track
    track = [track for track in mus.tracks if track.name == test_track][0]

    # convert to torch tensor
    audio = torch.tensor(track.audio.T, dtype=torch.float32)

    stft = model.STFT(n_fft=4096, n_hop=1024)
    spec = model.Spectrogram(power=1, mono=False)
    magnitude_spectrogram = spec(stft(audio[None, ...]))
    torch.save(magnitude_spectrogram, "Al James - Schoolboy Facination.spectrogram.pt")


if __name__ == "__main__":
    main()
