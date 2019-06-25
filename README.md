# Open-Unmix (PyTorch)

This repository contains the PyTorch implementation of __open-unmix__, a simple reference implementation for research and applications on music source separation.

* Design choices for _open unmix_ favored simplicity over performance, to promote clearness of the code. The rationale is to have the model serve as a __baseline__ with performance still meeting current standards. The results are comparable to those of `UHL1`, which obtained the best performance over all systems trained on MUSDB18 in the [SiSEC 2018 Evaluation campaign](https://sisec18.unmix.app)
* _open unmix_ uses as few external code as possible to allow researchers to reproduce existing results, quickly develop new architectures and add own user data for training and testing. As a result, we used framework specific implementations. We provide a fully working implementation for [pytorch], as well as guidelines for designing models in [tensorflow] and [nnabla].
* _open unmix_ provides a ready-to-use pre-trained model that allows users to separate pop music into individual stems.
* _open unmix_ is a community focused project, we therefore encourage the community to submit bug-fixes and comments. However, we are not looking for changes that only focused on improving the performance.

## Model Design

_Open Unmix_ is a recurrent model, based on a bi-directional deep [LSTM network](https://dl.acm.org/citation.cfm?id=1246450). The model learns to predict the target magnitude from a mixture spectrogram. The model is optimized in the magnitude domain and the actual separation is then done in a post-processing step involving a multichannel wiener filter. To perform separation into multiple sources, multiple models would have to trained for each particular target. While this makes the training less comfortable, it allows great flexibility to customize the training data for each target source. E.g. one might have more data for __mixture -> vocals__ than for __mixture -> bass__.

#### Input

The input of the model is a single or multichannel time domain signal tensor of shape `(nb_samples, nb_channels, nb_timesteps)`, where `nb_samples` is the batch size and `nb_timesteps` is the number of audio samples. The model processes spectrograms based on `torch.STFT` on the fly. Alternatively _open unmix_ also takes magnitude spectrograms
directly using `(nb_frames, nb_samples, nb_channels, nb_bins)`.

#### Normalization

We apply instance normalization in multiple stages of the model to make the training more robust to gain variation. At the output we de-normalize to match the gain of the target spectrogram. This is done by using global standardization parameters that are trainable.

#### Dimensionality reduction

The LSTM is not operating on the original input spectrogram resolution. Instead, in the first step after the normalization, the network learns to compresses the frequency axis of the model to reduce redundancy and make the model converge faster.

## Installation

For installation we recommend to use the [Anaconda](https://anaconda.org/) python distribution. To create a conda environment for _open-unmix_, simply run:

`conda env create -f environment-X.yml` where `X` is either [`cpu-linux`, `gpu-cuda10`, `cpu-osx`], depending on your system. For now, we do not support windows operating systems.

## Training

_open-unmix_ support standard pytorch `[Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)` classes. This repository comes with __three__ different datasets which cover a wide range of tasks. The dataset can be selected through a commandline parameter.

| Command line Argument      | Description                                                            | Default      |
|----------------------------|------------------------------------------------------------------------|--------------|
| `--dataset <str>`          | Name of the dataset `{musdb,aligned,unaligned}`                        | `musdb`      |
| `--root <str>`           | path to root of dataset                                                  | `None`       |
| `--output <str>`           | path to trained output model                                           | `./OSU`      |
| `--no_cuda`           | disable cuda even if available                                              | not set      |

#### `MUSDBDataset`

The [MUSDB18](https://sigsep.github.io/datasets/musdb.html) is the largest freely available dataset for professionally produced music tracks (~10h duration) of different styles. It comes with isolated `drums`, `bass`, `vocals` and `others` stems.

_MUSDB18_ contains two subsets: "train", composed of 100 songs, and "test", composed of 50 songs.
To train the _open unmix_ using the MUSDB18 dataset, you just need to set the following parameters:

```
python train.py --dataset musdb --root /data/musdb --target vocals
```

#### `AlignedDataset`

A dataset class that assumes track folders with aligned/paired targets. The dataset does not perform any mixing but directly uses the target files that are within the folder. The filenames would have to be identical for each track. E.g, for the first sample of the training, input could be `01/mixture.wav` and output could be `01/vocals.wav`.

To start training one need to run

```
python train.py --dataset aligned --root /data/data --input_file mixture.wav --output_file vocals.wav
```


#### `UnalignedDataset`

In this scenario, we assume sources to not be aligned. Instead of folders with individual audio tracks, the sources are organized in subfolders by the source name. In this scenario one could easily train a network to separate a target sounds from interfering sounds. For each sample, the data loader loads a random combination of target+interferer as the intput and performs a linear mixture of these. The output of the model is the target.

In the following example, we have three folders with a unmatched number of vocal sounds, music sounds and noise sounds. To perform vocals separation based on music+noise interferer, you would just need to run:

```
python train.py --dataset unaligned --root /data/data --target vocals --interferences music noises
```

### Model Parameters

Additional training parameters and their default values are listed below:

| Command line Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--epochs <int>`           | number of epochs to train                                                       | `1000`          |
| `--patience <int>`         | early stopping patience                                                         | `20`            |
| `--batch-size <int>`       | Batch size has influence on memory usage and performance of the LSTM layer      | `16`            |
| `--seq-dur <int>`          | Sequence duration in seconds of excerpts taken from the dataset.                | `5.0`           |
| `--lr <float>`             | learning rate                                                                   | `0.0001`        |
| `--seed <int>`             | Initial seed to set the random initialization                                   | `42`            |
| `--nfft <int>`             | STFT FFT window length in samples                                               | `4096`          |
| `--nhop <int>`             | STFT hop length in samples                                                      | `1024`          |
| `--seed <int>`             | Initial seed to set the random initialization                                   | `42`            |
| `--bandwidth <int>`        | maximum model bandwidth in Hertz                                                | `16000`         |
| `--nb-channels <int>`      | set number of channels for model (1 for mono, 2 for stereo)                     | `1`             |
| `--quiet`                  | disable print and progressbar during training                                   | not set         |

## Inference

Since PyTorch currently lacks an invertible STFT, the synthesis is performed in numpy. For the inference, we rely on an implementation of a multichannel Wiener filter, that is a very popular way of filtering multichannel audio for several applications, notably speech enhancement and source separation. The `norbert` module assumes to have some way of estimating power-spectrograms for all the audio sources (non-negative) composing a mixture. It then builds the filter that is appropriate for extracting those signals from a mixture, by optimally exploiting multichannel information (like in stereo signals). This is done in an iterative procedure called Expectation Maximization, where filtering and re-estimation of the parameters are iterated.

### Test using wav files

Assuming that a trained model is saved in `model_path` predicting separation results is simply done by running:

`python test.py model_path --input mixture.wav`

Note that `model_path` usually contains individual models for each target and performs separation using all models. E.g. if `model_path` contains `vocals` and `drums` models, two output files are generated.

### Evaluation using `museval`

To perform evaluation in comparison to other SISEC systems, you would need to install the `museval` package using

```
pip install museval
```

and then run the evaluation using

`python eval.py model_path --outdir /path/to/musdb/estimates --evaldir /path/to/museval/results`

### Parameters

The inference can be controlled with additional parameters that influence the performance of the separation

| Command line Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--niter <int>`           | umber of EM steps for the post-processing.`--niter 0` is equivalent to a ratio mask.                                                       | `0`          |
| `--alpha <float>`         |this value changes the exponent for the softmask $X^\alpha$. A smaller value allows for more X, whereas a larger value results in Y.                                                          | `1.0`            |
| `--logit`       | this option makes the softmask go through a logistic function, tending to binarize the filtering: this tends to reduce interference, but to augment distortion.      | not set            |


### Authors

Fabian-Robert Stöter, Antoine Liutkus

### License

MIT
