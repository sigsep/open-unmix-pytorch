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
import torch.utils
import data
import math


tqdm.monitor_interval = 0

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MUSMAG')

# which target do we want to train?
parser.add_argument('--target', type=str, default='vocals',
                    help='source target for musdb')

# I/O Parameters
parser.add_argument('--seq-dur', type=float, default=5.0)
parser.add_argument('--seq-hop', type=float, default=2.5)

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

dataset = data.RawDataset("~/repositories/getency/tencydb", seq_dur=args.seq_dur)
training_generator = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=8)
print("Sequence Length (samples): dataset.seq_len")
# math.ceil(len(args.seq_dur) / args.seq_hop) + 1
# samples = (nb_frames - 1) * hop
batch_size = 16

model = model.OSU(n_fft=2048, n_hop=1024, power=1).to(device)

optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()


def train(epoch):
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()
    end = time.time()

    for x, y in training_generator:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        Y_hat = model(x)
        Y = model.spec(y)
        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        losses.update(loss.item(), Y.size(1))
        print(losses.avg)
        # measure elapsed time
        data_time.update(time.time() - end)
    return losses.avg


def valid(gen):
    losses = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch in dataset:
            X = torch.tensor(batch['X'], dtype=torch.float32, device=device)
            Y = torch.tensor(batch['Y'], dtype=torch.float32, device=device)
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
