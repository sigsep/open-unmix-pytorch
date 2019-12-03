import pytest
import numpy as np
import data


def test_musdb():
    musdb = data.MUSDBDataset(
        download=True,
        samples_per_track=1,
        seq_duration=1.0
    )
    for x, y in musdb:
        print(x.mean())
