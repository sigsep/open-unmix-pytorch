import torch.nn.functional as F
import torch.optim as optim
import torch.nn
import argparse
import model
import torch
import time
from pathlib import Path
import tqdm
import json
import torch.utils.data
import utils
import data


tqdm.monitor_interval = 0

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MUSMAG')

# which target do we want to train?
parser.add_argument('--target', type=str, default='vocals',
                    help='source target for musdb')


parser.add_argument('--root', type=str, help='root path of dataset')
parser.add_argument('--is-wav', action='store_true', default=False,
                    help='flags wav version of the dataset')


# I/O Parameters
parser.add_argument('--seq-dur', type=float, default=1.0)

parser.add_argument('--output', type=str, default="OSU",
                    help='provide output path base folder name')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--patience', type=int, default=20,
                    help='early stopping patience (default: 20)')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, defaults to 1e-3')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# create output dir if not exist
target_path = Path(args.output, args.target)
target_path.mkdir(parents=True, exist_ok=True)

# use jpg or npy
torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

# # Alternative Data loading
# train_dataset = data.SourceFolderDataset(
#     root=Path(args.root, "train"),
#     seq_duration=args.seq_dur,
# )    

dataset_kwargs = {
    'root': args.root,
    'is_wav': args.is_wav,
    'seq_duration': args.seq_dur,
    'subsets': 'train',
    'target': args.target
}

train_dataset = data.MUSDBDataset(validation_split='train', **dataset_kwargs)
valid_dataset = data.MUSDBDataset(validation_split='valid', **dataset_kwargs)

train_sampler = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    **dataloader_kwargs
)
valid_sampler = torch.utils.data.DataLoader(
    valid_dataset, batch_size=1,
    **dataloader_kwargs
)

model = model.OSU(n_fft=2048, n_hop=1024, power=1).to(device)

optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()


def train(epoch):
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    model.train()
    end = time.time()

    for x, y in tqdm.tqdm(train_sampler):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        Y_hat = model(x)
        Y = model.transform(y)
        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
        data_time.update(time.time() - end)
    return losses.avg


def valid():
    losses = utils.AverageMeter()

    model.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            Y_hat = model(x)
            Y = model.transform(y)
            loss = F.mse_loss(Y_hat, Y)
            losses.update(loss.item(), x.size(1))
        return losses.avg


es = utils.EarlyStopping(patience=args.patience)
best_loss = 1000
t = tqdm.trange(1, args.epochs + 1)
train_losses = []
valid_losses = []
for epoch in t:
    train_loss = train(epoch)
    valid_loss = valid()
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
        'valid_loss_history': valid_losses,
    }

    with open(Path(target_path,  "output.json"), 'w') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))
