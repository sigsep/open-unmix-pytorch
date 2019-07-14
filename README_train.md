# Training Open-Unmix

## Datasets

_open-unmix_ uses standard PyTorch [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) classes. This repository comes with __five__ different datasets which cover a wide range of tasks and applications around source separation. The dataset can be selected through a command line parameter.

| Command line Argument      | Description                                                            | Default      |
|----------------------------|------------------------------------------------------------------------|--------------|
| `--dataset <str>`          | Name of the dataset `{musdb,aligned,'sourcefolder', 'trackfolder_var', 'trackfolder_fix'}`                        | `musdb`      |
| `--root <str>`           | path to root of dataset on disk.                                                  | `None`       |
| `--output <str>`           | path where to save the trained output model as well as checkpoints.                         | `./umx`      |
| `--no_cuda`           | disable cuda even if available                                              | not set      |

### `MUSDBDataset` (default)

The [MUSDB18](https://sigsep.github.io/datasets/musdb.html) and [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) are the largest freely available dataset for professionally produced music tracks (~10h duration) of different styles. They come with isolated `drums`, `bass`, `vocals` and `others` stems.

_MUSDB18_ contains two subsets: "train", composed of 100 songs, and "test", composed of 50 songs.
To train the vocal model with _open-unmix_ using the MUSDB18 dataset, you just need to set the following command:

```bash
python train.py --dataset musdb --root /data/musdb --target vocals
```

#### Parameters



### `AlignedDataset`

A dataset of that assumes multiple track folders where each track includes and input and an output file which directly corresponds to the the input and the output of the model.

The dataset does not perform any mixing but directly uses the target files that are within the folder. The filenames would have to be identical for each track. E.g, for the first sample of the training, input could be `1/mixture.wav` and output could be `1/vocals.wav`.

This dataset is the most basic of all datasets provided here, due to the least amount of
preprocessing, it is also the fastest option, however,
it lacks any kind of source augmentations or custum mixing.

Typical use cases:

* Source Separation (Mixture -> Target)
* Denoising (Noisy -> Clean)
* Bandwidth Extension (Low Bandwidth -> High Bandwidth)

#### File Structure

```
data/train/01/mixture.wav --> input
data/train/01/vocals.wav ---> output
```

#### Example

```bash
python train.py --dataset aligned --root /data/data --input_file mixture.wav --output_file vocals.wav
```

### `SourceFolderDataset`

A dataset of that assumes folders of sources,
instead of track folders. This is a common
format for speech and environmental sound datasets
such das DCASE. For each source a variable number of
tracks/sounds is available, therefore the dataset is unaligned by design.

In this scenario one could easily train a network to separate a target sounds from interfering sounds. For each sample, the data loader loads a random combination of target+interferer as the input and performs a linear mixture of these. The output of the model is the target.

#### File structure

```
train/vocals/track11.wav -----------------\
train/drums/track202.wav  (interferer1) ---+--> input
train/bass/track007a.wav  (interferer2) --/

train/vocals/track11.wav ---------------------> output
```

#### Parameters

...

#### Example

```bash
python train.py --dataset sourcefolder --root /data --target-dir vocals --interferer-dirs carnoise windnoise --ext .ogg --nb-train-samples 1000
```

### `FixedSourcesTrackFolderDataset`

A dataset of that assumes audio sources to be stored
in track folder where each track has a fixed number of sources. For each track the users specifies the target file-name (`target_file`) and a list of interferences files (`interferer_files`).
A linear mix is performed on the fly by summing the target and the interferers up.

Due to the fact that all tracks comprise the exact same set of sources, the random track mixing augmentation technique can be used, where sources from different tracks are mixed together. Setting `random_track_mix=True` results in an unaligned dataset.
When random track mixing is enabled, we define an epoch as when the the target source from all tracks has been seen and only once with whatever interfering sources has randomly been drawn.

This dataset is recommended to be used for small/medium size for example like the MUSDB18 or other custom source separation datasets.

#### File structure

```sh
train/1/vocals.wav ---------------\
train/1/drums.wav (interferer1) ---+--> input
train/1/bass.wav -(interferer2) --/

train/1/vocals.wav -------------------> output
```

#### Example

```
python train.py  --root /data --dataset trackfolder_fix --target-file vocals.flac --interferer-files bass.flac drums.flac other.flac
```

### `VariableSourcesTrackFolderDataset`

A dataset of that assumes audio sources to be stored in track folder where each track has a _variable_ number of sources. The users specifies the target file-name (`target_file`) and the extension of sources to used for mixing. A linear mix is performed on the fly by summing all sources in a track folder.

Since the number of sources differ per track, while target is fixed, a random track mix augmentation cannot be used.

Also make sure, that you do not provide the mixture file among the sources!

#### File structure

```sh
train/1/vocals.wav --> input target   \
train/1/drums.wav --> input target     |
train/1/bass.wav --> input target    --+--> input
train/1/accordion.wav --> input target |
train/1/marimba.wav --> input target  /

train/1/vocals.wav -----------------------> output
```

#### Example


```
python train.py --root /data --dataset trackfolder_var --target-file vocals.flac --ext .wav
```

## Training and Model Parameters

Additional training parameters and their default values are listed below:

| Command line Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--epochs <int>`           | number of epochs to train                                                       | `1000`          |
| `--patience <int>`         | early stopping patience                                                         | `250`            |
| `--batch-size <int>`       | Batch size has influence on memory usage and performance of the LSTM layer      | `16`            |
| `--seq-dur <int>`          | Sequence duration in seconds of excerpts taken from the dataset.                | `6.0`           |
| `--lr <float>`             | learning rate                                                                   | `0.0001`        |
| `--seed <int>`             | Initial seed to set the random initialization                                   | `42`            |
| `--nfft <int>`             | STFT FFT window length in samples                                               | `4096`          |
| `--nhop <int>`             | STFT hop length in samples                                                      | `1024`          |
| `--seed <int>`             | Initial seed to set the random initialization                                   | `42`            |
| `--bandwidth <int>`        | maximum model bandwidth in Hertz                                                | `16000`         |
| `--nb-channels <int>`      | set number of channels for model (1 for mono, 2 for stereo)                     | `1`             |
| `--quiet`                  | disable print and progress bar during training                                   | not set         |

## Output Files

### Loss curves

### Arguments