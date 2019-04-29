# Open-Unmix (PyTorch)

This repository contains the PyTorch implementation of __open-unmix__, a simple reference implementation for research and applications on music source separation.

Design decisions:

* _open unmix_ is __not__ state-of-the-art but instead should serve as baseline. The results of this baseline are comparable to the results of [1], as evaluated in the [SiSEC 2018 Evaluation campaign](https://sisec18.unmix.app)
* _open unmix_ uses as little external code as possible to allow researchers to reproduce existing results, quickly develop new architectures and add own user data for training and testing. As a result, we used framework specific implementations. We provide a model for [pytorch], [tensorflow] and [nnabla], however the performance is not identical due to the differences between frameworks.
* _open unmix_ provides a ready-to-use pre trained model that allows users to separate pop music into individual stems.
* _open unmix_ is community focused project, we therefore encourage the community to submit bug-fixes and comments. However, we are not looking for changes that only focused on improving the performance. 

## Model

Open Unmix is a recurrent model, based on the LSTM network. The model is learning to predict the target magnitude from a mixture spectrogram. 

### Layers

|               | Input                                                                           | Shape      |
|---------------|---------------------------------------------------------------------------------|------------|
| Input         | Time domain audio or magnitude spectrogram (single or multichannel)             |            |
| Output        | magnitude spectrogram                                                           |            |
| Normalization | Instance Norm                                                                   |            |
| De-Normalization | Mean/Scale Denormalisation                                                   |            |
| Model         | 3 Layer LSTM with time distributed dense layers to compress/decompress the data |            |


## Installation

For installation we recommend to use the [Anaconda](https://anaconda.org/) python distribution. To create a conda environment for _open-unmix_, simply run:

`conda env create -f environment-X.yml` where `X` is either [`cpu-linux`, `gpu-cuda10`, `cpu-osx`]

## Training

_open-unmix_ support standard pytorch `[Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)` classes. This repository comes with __three__ different datasets which cover a wide range of tasks.

### Datasets
...

#### `MUSDBDataset`

The [MUSDB18](https://sigsep.github.io/datasets/musdb.html) is the largest freely available dataset for professionally produced music tracks (~10h duration) of different styles. It comes with isolated `drums`, `bass`, `vocals` and `others` stems.

_MUSDB18_ contains two subsets: "train", composed of 100 songs, and "test", composed of 50 songs. To train

To train the _open unmix_ using the MUSDB18 dataset, you just need to set the default parameters

```
python train.py --dataset musdb
```

#### `AlignedDataset`

...

#### `UnalignedDataset`

A dataset of that assumes sources to be unaligned,
organized in subfolders with the name of sources


```python
dataset = data.UnalignedDataset(target='vocals', interferences=['noise', 'musics'], ...)
```

Example:

```python
["noises/10923.wav", "music/asjdk.wav", "vocals/1.wav"]
# this would create linear mixture of noise/10923.wav, /music/asjdk.wav and vocals/1.wav
# for the input of the model, and vocals/1.wav as the output of the model
```

Scales to a large amount of audio data.
Uses pytorch' index based sample access

### Model Parameters

```
 --epochs N            number of epochs to train (default: 1000)
 --patience PATIENCE   early stopping patience (default: 20)
 --batch_size BATCH_SIZE defaults to 16
 --seq-dur SEQ_DUR     LSTM sequence duration, defaults to 5.0 seconds
 --lr LR               learning rate, defaults to 1e-3
 --seed S              random seed (default: 1)
 --nfft NFFT           fft size, defaults to 4096
 --nhop NHOP           hop size, defaults to 1024
 --bandwidth BANDWIDTH maximum model bandwidth in herz, defaults to 16000
 --nb-channels NB_CHANNELS
                       set number of channels for model (1, 2)
 --quiet               less verbose during training
```

## Inference

### Output Parameters

#### `--niter`

We rely on an implementation of a multichannel Wiener filter, that is a very popular way of filtering multichannel audio for several applications, notably speech enhancement and source separation. The `norbert` module assumes to have some way of estimating power-spectrograms for all the audio sources (non-negative) composing a mixture. It then builds the filter that is appropriate for extracting those signals from a mixture, by optimally exploiting multichannel information (like in stereo signals). This is done in an iterative procedure called Expectation Maximization, where filtering and re-estimation of the parameters are iterated. `--niter` sets the number of EM steps for the postprocessing. _open unmix_ defaults to `--niter 0` which is just computing a simple ratio mask filter.

#### `--logit`

...

#### `--alpha`

...

### Test using wav files

`python test.py {A} --input mixture.wav`

### Evaluation using `museval`

`python eval.py {A} --outdir {B} --evaldir {C}`


### Pre-trained model

t.b.a.


### Citation

* [1] S. Uhlich, M. Porcu, F. Giron, M. Enenkl, T. Kemp, N. Takahashi and Y. Mitsufuji. “Improving music source separation based on deep neural networks through data augmentation and network blending”, Proc. ICASSP, 2017.

### License

MIT