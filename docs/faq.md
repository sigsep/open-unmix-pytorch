# Frequently Asked Questions

## Separating tracks crashes because it used too much memory

First, separating an audio track into four separation models `vocals`, `drums`, `bass` and `other` is requires a significant amount of RAM to load all four separate models.
Furthermore, another computationally important step in the separation is the post-processing of the `norbert` package, controlled by the parameter `niter`. 
For faster and less memory intensive inference (at the expense of separation quality) it is advised to use `niter 0`.
Another way to improve performance is to apply separation on smaller excerpts using the `start` and `duration`, arguments. We suggest to only perform separation of ~30s stereo excerpts on machines with less 8GB of memory.

## Why is the training so slow?

In the default configuration using the stems dataset, yielding a single batch from the dataset is very slow. This is a known issue of decoding mp4 stems since native decoders for pytorch or numpy are not available.

There are two ways to speed up the training

### Increase the number of workers

The default configuration does not use multiprocessing to yield the batches. You can increase the number of workers using the `--nb-workers k` configuration. E.g. `k=8` workers batch loading can get down to 1 batch per second.

### Use WAV instead of MP4

Convert the MUSDB18 dataset to wav using the builtin `musdb` cli tool

```
musdbconvert path/to/musdb-stems-root path/to/new/musdb-wav-root
```

or alternatively use the [MUSDB18-HQ](https://zenodo.org/record/3338373) dataset that is already stored and distributed as WAV files. Note that __if you want to compare to SiSEC 2018 participants, you should use the standard (Stems) MUSDB18 dataset and decode it to WAV, instead.__

Training on wav files can be launched using the `--is-wav` flag:

```
python train.py --root path/to/musdb18-wav --is-wav --target vocals
```

This will get you down to 0.6s per batch on 4 workers, likely hitting the bandwidth of standard hard-drives. It can be further improved using an SSD, which brings it down to 0.4s per batch on a GTX1080Ti which this leads to 95% GPU utilization. thus data-loading will not be the bottleneck anymore.

## Can I use the pre-trained models `umx` or `umxhq` without torchhub?

for some reason the torchub automatic download might not work and you want to download the files offline and use them. For that you can download [umx](https://zenodo.org/record/3340804) or [umxhq](https://zenodo.org/record/3267291) from Zenodo and create a local folder of your choice (e.g. `umx-weights`) where the model is stored in a flat file hierarchy:

```
umx-weights/vocals-*.pth
umx-weights/drums-*.pth
umx-weights/bass-*.pth
umx-weights/other-*.pth
umx-weights/vocals.json
umx-weights/drums.json
umx-weights/bass.json
umx-weights/other.json
```

Test and eval can then be started using:

```bash
python test.py --model umx-weights --input test.wav
```
