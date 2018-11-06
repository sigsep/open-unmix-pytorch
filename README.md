# Open-Unmix-PyTorch

## Installation

### Manual installation using Pip

Make sure to install pytorch for your system (either CPU or GPU). Then install the additional pip requirements using

`pip install pescador tqdm torchvision`

Furthermore some of the packages are currently not available on pypi, therefore, please install manually from github

```
pip install git+https://github.com/sigsep/sigsep-mus-db#egg=musdb
pip install git+https://github.com/sigsep/norbert#egg=norbert
```

### Pipenv

Install [pipenv](https://pipenv.readthedocs.io/en/latest/) via `pip install pipenv`, then install the requirements via `pipenv install`.

### Anaconda

`conda env create -f environment-X.yml` where `X` is either [`cpu`, `gpu-cuda92`, `linux`]

## Training

### Train with precomputed 30s jpg images

[musdb](https://github.com/sigsep/sigsep-mus-db) includes pre-computed magnitudes of the [MUSDB 30s previews](https://zenodo.org/record/1256003). These magnitudes are automatically downloaded to the folder set by the `data-dir` option (defaults to `./data`). To start the training of the vocals model, you just run:

`python train.py --target vocals`

### Train with raw npy spectrograms

Due to quantization of the magnitudes, training with jpg images produce slightly different models.
To avoid this, one can train using pre-computed STFT tensors, saved in the `.npy` format. Due to its file size, we don't offer this to download. Instead, please run the dataset creation script (`create_dataset.py`) from the main folder. When this is finished, the training can be started using:

`python train.py --target vocals --data-dir musmag-npy --data-type .npy`

## Test

t.b.a.

### MUSDB Test

t.b.a.

### Evaluation using `museval`

t.b.a.

### Pretrained model

t.b.a.
