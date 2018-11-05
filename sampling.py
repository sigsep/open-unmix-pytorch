import numpy as np
import pescador
import random


def excerpt_generator(
    dataset, idx,
    ex_length=256, ex_hop=256,
    shuffle=True, data_type='.npy',
    seed=42
):
    X, Y = dataset[idx]
    nb_frames, nb_bins, nb_channels = X.shape

    # get all t indices and shuffle, make sure that excerpt is shorter than
    # number of frames
    if ex_length < nb_frames:
        steps = np.arange(0, nb_frames - ex_length, ex_hop)
    else:
        steps = [0]
        ex_length = nb_frames

    if shuffle:
        # shuffle t indices in place
        np.random.seed(seed)
        np.random.shuffle(steps)

    for s in steps:
        cur_X = X[s:s+ex_length, ...]
        cur_Y = Y[s:s+ex_length, ...]
        # dequantize from 8bit int to float
        if data_type == '.jpg':
            cur_X = np.exp(cur_X.astype(np.float) / (2 ** 8 - 1)) - 1
            cur_Y = np.exp(cur_Y.astype(np.float) / (2 ** 8 - 1)) - 1
        yield dict(X=cur_X, Y=cur_Y)


def mixmux(
    dataset,
    batch_size,
    ex_length=256,
    ex_hop=256,
    shuffle=True,
    seed=42,
    active_streamers=100,
    data_type='.npy'
):
    streams = []
    for idx in range(len(dataset)):
        s = pescador.Streamer(
            excerpt_generator,
            dataset, idx,
            ex_length=ex_length,
            ex_hop=ex_hop,
            shuffle=shuffle,
            seed=seed,
            data_type=data_type
        )
        streams.append(s)

    if not shuffle:
        mux = pescador.ChainMux(streams, mode='exhaustive')
    else:
        random.seed(seed)
        # shuffle the streams, just in case
        random.shuffle(streams)

        mux = pescador.StochasticMux(
            streams,
            active_streamers,
            rate=None,
            mode='exhaustive',
            random_state=seed
        )

    # get `batch_size` samples from the mux
    batches = pescador.Streamer(pescador.buffer_stream, mux, batch_size)

    return batches
