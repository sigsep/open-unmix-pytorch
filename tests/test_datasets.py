import pytest
import numpy as np


from umx import data


def test_musdb():
    musdb = data.MUSDBDataset(
        download=True,
        samples_per_track=1,
        seq_duration=1.0
    )
    for x, y in musdb:
        assert x.shape[-1] == 44100


def test_trackfolder_fix():
    train_dataset = data.FixedSourcesTrackFolderDataset(
        split='train',
        seq_duration=1.0,
        root="TrackfolderDataset",
        sample_rate=8000,
        target_file='1.wav',
        interferer_files=['2.wav', '3.wav', '4.wav'],
    )
    for x, y in train_dataset:
        assert x.shape[-1] == 8000


def test_trackfolder_var():
    train_dataset = data.VariableSourcesTrackFolderDataset(
        split='train',
        seq_duration=1.0,
        root="TrackfolderDataset",
        sample_rate=8000,
        target_file='1.wav',
    )
    for x, y in train_dataset:
        assert x.shape[-1] == 8000


def test_sourcefolder():
    train_dataset = data.SourceFolderDataset(
        split='train',
        seq_duration=1.0,
        root="TrackfolderDataset",
        sample_rate=8000,
        target_dir='1',
        interferer_dirs=['2', '3'],
        ext='.wav',
        nb_samples=20
    )
    for k in range(len(train_dataset)):
        x, y = train_dataset[k]
        assert x.shape[-1] == 8000
