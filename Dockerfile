FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsox-fmt-all \
    sox \
    libsox-dev

WORKDIR /workspace

RUN conda install ffmpeg -c conda-forge
RUN pip install musdb>=0.4.0

RUN pip install openunmix['stempeg']

ENTRYPOINT ["umx"]