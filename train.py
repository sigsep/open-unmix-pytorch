import argparse
import model
import data
import torch
import time
from pathlib import Path
import tqdm
import json
import utils
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy


tqdm.monitor_interval = 0

parser = argparse.ArgumentParser(description='Open Unmix Trainer')

# which target do we want to train?
parser.add_argument('--target', type=str, default='vocals',
                    help='source target for musdb')

# Dataset paramaters
parser.add_argument('--dataset', type=str, default="musdb",
                    choices=[
                        'musdb', 'aligned', 'sourcefolder',
                        'trackfolder_var', 'trackfolder_fix'
                    ],
                    help='Name of the dataset.')
parser.add_argument('--root', type=str, help='root path of dataset')
parser.add_argument('--output', type=str, default="OSU",
                    help='provide output path base folder name')

# Trainig Parameters
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--patience', type=int, default=140,
                    help='early stopping patience (default: 20)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, defaults to 1e-3')
parser.add_argument('--lr-decay-patience', type=int, default=70,
                    help='lr decay patience for plateaeu scheduler')
parser.add_argument('--lr-decay-gamma', type=float, default=0.1,
                    help='gamma of learning rate scheduler decay')
parser.add_argument('--weight-decay', type=float, default=0.00001,
                    help='gamma of learning rate scheduler decay')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

# Model Parameters
parser.add_argument('--seq-dur', type=float, default=6.0,
                    help='Sequence duration in seconds'
                    'value of <=0.0 will use full/variable length')
parser.add_argument('--unidirectional', action='store_true', default=False,
                    help='Use unidirectional LSTM instead of bidirectional')
parser.add_argument('--nfft', type=int, default=4096,
                    help='STFT fft size and window size')
parser.add_argument('--nhop', type=int, default=1024,
                    help='STFT hop size')
parser.add_argument('--hidden-size', type=int, default=512,
                    help='hidden size parameter of FC bottleneck layers')
parser.add_argument('--bandwidth', type=int, default=16000,
                    help='maximum model bandwidth in herz')
parser.add_argument('--nb-channels', type=int, default=2,
                    help='set number of channels for model (1, 2)')
parser.add_argument('--nb-workers', type=int, default=0,
                    help='Number of workers for dataloader.'
                    'Can be >0 e.g. when loading wav files')

# Misc Parameters
parser.add_argument('--quiet', action='store_true', default=False,
                    help='less verbose during training')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args, _ = parser.parse_known_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}


repo_dir = os.path.abspath(os.path.dirname(__file__))
repo = Repo(repo_dir)
commit = repo.head.commit.hexsha[:7]

# use jpg or npy
torch.manual_seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

train_dataset, valid_dataset, args = data.load_datasets(parser, args)

# create output dir if not exist
target_path = Path(args.output)
target_path.mkdir(parents=True, exist_ok=True)

train_sampler = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    **dataloader_kwargs
)
valid_sampler = torch.utils.data.DataLoader(
    valid_dataset, batch_size=1,
    **dataloader_kwargs
)

input_scaler = sklearn.preprocessing.StandardScaler()
output_scaler = sklearn.preprocessing.StandardScaler()
spec = torch.nn.Sequential(
    model.STFT(n_fft=args.nfft, n_hop=args.nhop),
    model.Spectrogram(mono=True)
)

train_dataset_scaler = copy.deepcopy(train_dataset)
train_dataset_scaler.samples_per_track = 1
train_dataset_scaler.augmentations = None
for x, y in tqdm.tqdm(train_dataset_scaler, disable=args.quiet):
    X = spec(x[None, ...])
    input_scaler.partial_fit(np.squeeze(X))

if not args.quiet:
    print("Computed global average spectrogram")

# set inital input scaler values
safe_input_scale = np.maximum(
    input_scaler.scale_,
    1e-4*np.max(input_scaler.scale_)
)

max_bin = utils.bandwidth_to_max_bin(
    train_dataset.sample_rate, args.nfft, args.bandwidth
)

unmix = model.OpenUnmix(
    input_mean=input_scaler.mean_,
    input_scale=safe_input_scale,
    nb_channels=args.nb_channels,
    hidden_size=args.hidden_size,
    n_fft=args.nfft,
    n_hop=args.nhop,
    max_bin=max_bin,
).to(device)

optimizer = torch.optim.Adam(
    unmix.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=args.lr_decay_gamma,
    patience=args.lr_decay_patience,
    cooldown=10
)


def train():
    losses = utils.AverageMeter()
    unmix.train()

    for x, y in tqdm.tqdm(train_sampler, disable=args.quiet):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        Y_hat = unmix(x)
        Y = unmix.transform(y)
        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
    return losses.avg


def valid():
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            Y_hat = unmix(x)
            Y = unmix.transform(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        return losses.avg


es = utils.EarlyStopping(patience=args.patience)
t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
train_losses = []
valid_losses = []
train_times = []
best_epoch = 0

for epoch in t:
    end = time.time()
    train_loss = train()
    valid_loss = valid()
    scheduler.step(valid_loss)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    t.set_postfix(
        train_loss=train_loss, val_loss=valid_loss
    )

    stop = es.step(valid_loss)

    if valid_loss == es.best:
        best_epoch = epoch

    utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': unmix.state_dict(),
            'best_loss': es.best,
            'optimizer': optimizer.state_dict(),
        },
        is_best=valid_loss == es.best,
        path=target_path,
        target=args.target
    )

    # save params
    params = {
        'epochs_trained': epoch,
        'args': vars(args),
        'best_loss': es.best,
        'best_epoch': best_epoch,
        'train_loss_history': train_losses,
        'valid_loss_history': valid_losses,
        'train_time_history': train_times,
        'commit_hash': commit
    }

    with open(Path(target_path,  args.target + '.json'), 'w') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))

    train_times.append(time.time() - end)

    if stop:
        print("Apply Early Stopping")
        break
