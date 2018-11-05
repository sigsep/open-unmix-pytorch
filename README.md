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

### Train with MUSDB JPGs

`python train.py --vocals`

### Train with npy spectrograms

`python train.py --target vocals --data-dir path_to_musdb_dir --data-type .npy`
