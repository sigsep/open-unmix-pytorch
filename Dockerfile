FROM pytorch/pytorch

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsox-fmt-all \
    sox \
    libsox-dev

WORKDIR /workspace

# install torchaudio from source
RUN git clone https://github.com/pytorch/audio.git pytorchaudio && cd pytorchaudio && python setup.py install