# Extending Open-Unmix

![](https://docs.google.com/drawings/d/e/2PACX-1vQ1WgVU4PGeEqTQ26j-2RbwaN9ZPlxabBI5N7mYqOK66VjT96UmT9wAaX1s6u6jDHe0ARfAo9E--lQM/pub?w=1918&h=703)
One of the key aspects of _Open-Unmix_ is that it was made to be easily extensible and thus is a good starting point for new research on music source separation. In fact, the open-unmix training code is based on the [pytorch MNIST example](https://github.com/pytorch/examples/blob/master/mnist/main.py). In this document we provide a short overview of ways to extend open-unmix.

## Code Structure

* `data.py` includes several torch datasets that can all be used to train _open-unmix_.
* `train.py` includes all code that is necessary to start a training.
* `model.py` includes the open-unmix torch modules.
* `test.py` includes code to predict/unmix from audio files.
* `eval.py` includes all code to run the objective evaluation using museval on the MUSDB18 dataset.
* `utils.py` includes additional tools like audio loading and metadata loading.

## Provide a custom dataset

Users of open-unmix that have their own datasets and could not fit one of our predefined datasets might want to implement or use their own `torch.utils.data.Dataset` to be used for the training. Such a modification is very simple since our dataset.

### Template Dataset

In case you want to create your own dataset we provide a template for the open-unmix API. You can use our efficient torchaudio or libsndfile based `load_audio` audio loaders or just use your own files. Since currently (pytorch<=1.1) is using index based datasets (instead of iterable based datasets), the best way to load audio is to assign the index to one audio track. However, there are possible applications where the index is ignored and the `__len__()` method just returns arbitrary number of samples.

```python
from utils import load_audio, load_info
class TemplateDataset(UnmixDataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, root, split='train', sample_rate=44100, seq_dur=None):
        """Initialize the dataset
        """
        self.root = root
        self.tracks = get_tracks(root, split)

    def __getitem__(self, index):
        """Returns a time domain audio example
        of shape=(channel, sample)
        """
        path = self.tracks[index]
        x = load_audio(path)
        y = load_audio(path)
        return x, y

    def __len__(self):
        """Return the number of audio samples"""
        return len(self.tracks)
```

## Provide a custom model

We think that recurrent models provide the best trade-off between good results, fast training and flexibility of training due to its ability to learn from arbitrary durations of audio and different audio representations. If you want to try different models you can easily build upon our model template below:

### Template Spectrogram Model

```python
from model import Spectrogram, STFT
class Model(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        nb_channels=2,
        input_is_spectrogram=False,
        sample_rate=44100.0,
    ):
        """
        Input:  (batch, channel, sample)
            or  (frame, batch, channels, frequency)
        Output: (frame, batch, channels, frequency)
        """

        super(OpenUnmix, self).__init__()

    def forward(self, mix):
        # transform to spectrogram on the fly
        X = self.transform(mix)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # transform X to estimate
        # ....

        return X
```

## Jointly train targets

We designed _open-unmix_ so that the training of multiple targets is handled in separate models. We think that this has several benefits such as:

* single source models can leverage unbalanced data where for each source different size of training data is available/
* training can easily distributed by training multiple models on different nodes in parallel.
* at test time the selection of different models can be adjusted for specific applications.

However, we acknowledge the fact that there might be reasons to train a model jointly for all sources to improve the separation performance. These changes can easily be made in _open-unmix_ with the following modifications based the way how pytorch handles single-input-multiple-outputs models.

### 1. Extend `data.py`

The dataset should be able to yield a list of tensors (one for each target): E.g. the `musdb` dataset can be extended with:

```python
y = [stems[ind] for ind, _ in enumerate(self.targets)]
```

### 2. Extend `model.py`

The _open-unmix_ model can be left unchanged but instead a "supermodel" can be added that joins the forward paths of all targets:

```python
class OpenUnmixJoint(nn.Module):
    def __init__(
        self,
        targets,
        *args, **kwargs
    ):
        super(OpenUnmixJoint, self).__init__()
        self.models = nn.ModuleList(
            [OpenUnmix(*args, **kwargs) for target in targets]
        )

    def forward(self, x):
        return [model(x) for model in self.models]
```

### 3. Extend `train.py`

The training should be updated so that the total loss is an aggregation of the individual target losses. For the mean squared error, the following modifications should be sufficient:

```python
criteria = [torch.nn.MSELoss() for t in args.targets]
# ...
for x, y in tqdm.tqdm(train_sampler, disable=args.quiet):
    x = x.to(device)
    y = [i.to(device) for i in y]
    optimizer.zero_grad()
    Y_hats = unmix(x)
    loss = 0
    for Y_hat, target, criterion in zip(Y_hats, y, criteria):
        loss = loss + criterion(Y_hat, unmix.models[0].transform(target))
```

## End-to-End time-domain models

If you want to evaluate models that work in the time domain such as WaveNet or WaveRNN, the training code would have to modified. Instead of spectrogram output `Y` the output is simply a time domain signal `y` that can directly be compared with `x`. E.g. going from:

```python
Y_hat = unmix(x)
Y = unmix.transform(y)
loss = criterion(Y_hat, Y)
```

to:

```python
y_hat = unmix(x)
loss = criterion(y_hat, y)
```

Inference, in that case, would then have to drop the spectral wiener filter and instead directly save the time domain signal (and maybe its residual):

```python
est = unmix(audio_torch).cpu().detach().numpy()
estimates[target] = est[0].T
estimates['residual'] = audio - est[0].T
```
