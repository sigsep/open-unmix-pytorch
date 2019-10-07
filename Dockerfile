FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsox-fmt-all \
    sox \
    libsox-dev

WORKDIR /workspace

# install torchaudio from source
RUN git clone https://github.com/pytorch/audio.git pytorchaudio && cd pytorchaudio && python setup.py install

COPY model.py /workspace
COPY data.py /workspace
COPY train.py /workspace
COPY utils.py /workspace
COPY eval.py /workspace
COPY test.py /workspace
COPY hubconf.py /workspace

RUN conda install tqdm=4.28 ffmpeg resampy -c conda-forge

RUN pip install musdb>=0.3.0
RUN pip install norbert>=0.2.0

ENTRYPOINT ["python", "test.py"]