import torch.nn.functional as F
import torch.optim as optim
import torch.nn
import argparse
import model
import torch
import time
from pathlib import Path
import tqdm
import numpy as np
import json
import sampling
import torch_utils as utils
import musdb


tqdm.monitor_interval = 0

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MUSMAG')

# which target do we want to train?
parser.add_argument('--target', type=str, default='vocals',
                    help='source target for musdb')

# I/O Parameters
parser.add_argument('--data-dir', type=str, default='data',
                    help='set data root dir')
parser.add_argument('--output', type=str, default="OSU",
                    help='provide output path base folder name')
parser.add_argument('--data-type', type=str, default=".jpg")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--patience', type=int, default=20,
                    help='early stopping patience (default: 20)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, defaults to 1e-3')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()

# create output dir if not exist
target_path = Path(args.output, args.target)
target_path.mkdir(parents=True, exist_ok=True)

# use jpg or npy
torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")


# Fixed Parameters
valid_tracks = [
    'Aimee Norwich - Child',
    'BigTroubles - Phantom',
    'Hollow Ground - Left Blind',
    'Music Delta - Disco',
    'Music Delta - Beatles',
    'The So So Glos - Emergency'
    'Port St Willow - Stay Even',
    'Voelund - Comfort Lives In Belief'
]

batch_size = 16
excerpt_length = 256
excerpt_hop = 256

d_train = musdb.MAG(
    root_dir=args.data_dir,
    target=args.target,
    valid_tracks=valid_tracks,
    download=args.data_type == '.jpg',
    valid=False,
    scale=True,
    in_memory=False,
    data_type=args.data_type
)

d_valid = musdb.MAG(
    root_dir=args.data_dir,
    target=args.target,
    download=args.data_type == '.jpg',
    valid_tracks=valid_tracks,
    valid=True,
    in_memory=False,
    data_type=args.data_type
)

nb_frames, nb_features, nb_channels = d_train[0][0].shape

train_mux = sampling.mixmux(
    d_train,
    batch_size,
    ex_length=excerpt_length,
    ex_hop=excerpt_hop,
    shuffle=True,
    data_type=args.data_type
)

# do not use overlap for validation
valid_mux = sampling.mixmux(
    d_valid,
    batch_size,
    ex_length=excerpt_length,
    ex_hop=excerpt_length,
    shuffle=False,
    data_type=args.data_type
)

model = model.OSU(
    nb_features=nb_features, nb_frames=excerpt_length,
    output_mean=d_train.output_scaler.mean_
).to(device)

optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()


def train(epoch):
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()
    end = time.time()

    for _, batch in enumerate(tqdm.tqdm(train_mux())):
        X = np.transpose(np.copy(batch['X']), (1, 0, 3, 2))
        Y = np.transpose(np.copy(batch['Y']), (1, 0, 3, 2))
        X = torch.tensor(X, dtype=torch.float32, device=device)
        Y = torch.tensor(Y, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        Y_hat = model(X)

        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        losses.update(loss.item(), X.size(1))
        # measure elapsed time
        data_time.update(time.time() - end)
    return losses.avg


def valid(gen):
    losses = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(gen):
            X = np.transpose(np.copy(batch['X']), (1, 0, 3, 2))
            Y = np.transpose(np.copy(batch['Y']), (1, 0, 3, 2))
            X = torch.tensor(X, dtype=torch.float32, device=device)
            Y = torch.tensor(Y, dtype=torch.float32, device=device)

            Y_hat = model(X)
            loss = F.mse_loss(Y_hat, Y)
            losses.update(loss.item(), X.size(1))

        return losses.avg, X, Y, Y_hat


es = utils.EarlyStopping(patience=args.patience)
best_loss = 1000
t = tqdm.trange(1, args.epochs + 1)
train_losses = []
valid_losses = []
for epoch in t:
    train_loss = train(epoch)
    valid_loss, X, Y, Y_hat = valid(valid_mux)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

    is_best = valid_loss < best_loss
    best_loss = min(valid_loss, best_loss)

    utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
        },
        is_best,
        target_path,
        args.target
    )

    if es.step(valid_loss):
        print("Apply Early Stopping")
        break

    # save params
    params = {
        'epochs_trained': epoch,
        'args': vars(args),
        'best_loss': str(best_loss),
        'train_loss_history': train_losses,
        'nb_frames': excerpt_length,
        'nb_features': nb_features,
        'nb_channels': nb_channels,
        'valid_loss_history': valid_losses,
    }

    with open(Path(target_path,  "output.json"), 'w') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))
