python -m pip install -e .['stempeg'] --quiet

# run umx on url
coverage run -a `which umx` https://samples.ffmpeg.org/A-codecs/wavpcm/test-96.wav --audio-backend stempeg
coverage run -a `which umx` https://samples.ffmpeg.org/A-codecs/wavpcm/test-96.wav --audio-backend stempeg --outdir out --niter 0
