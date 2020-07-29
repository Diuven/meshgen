ARG PYTORCH="1.4"
ARG CUDA="10.1"
ARG CUDNN="7"

# PYTHON
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV LC_ALL=C.UTF-8
ENV TERM=xterm-256color

RUN apt-get update && apt-get install -y --no-install-recommends \
    make git unzip wget bzip2 xz-utils \
    openssl build-essential \
    libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg

RUN python3 -m pip install \
    omegaconf==2.0.0 tqdm==4.48.0 imageio Pillow scikit-image open3d \
    pytorch-lightning==0.8.5 opencv-python wandb tensorboardx

ARG TORCH="1.4.0"
ARG CUDA="cu101"

RUN python3 -m pip install -f https://pytorch-geometric.com/whl/torch-${TORCH}.html \
    torch-scatter==latest+${CUDA} \
    torch-sparse==latest+${CUDA} \
    torch-cluster==latest+${CUDA} \
    torch-spline-conv==latest+${CUDA}

RUN python3 -m pip install torch-geometric

RUN wget https://anaconda.org/pytorch3d/pytorch3d/0.2.0/download/linux-64/pytorch3d-0.2.0-py37_cu101_pyt14.tar.bz2 \
    && conda install ./pytorch3d-0.2.0-py37_cu101_pyt14.tar.bz2 \
    && rm ./pytorch3d-0.2.0-py37_cu101_pyt14.tar.bz2

# RUN conda install -c pytorch3d pytorch3d
