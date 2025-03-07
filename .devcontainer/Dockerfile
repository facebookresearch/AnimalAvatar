FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and other necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    unzip \
    tar \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Set Python 3.10 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo 'source /opt/venv/bin/activate' >> ~/.bashrc

# pytorch
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# pytorch3d
# RUN curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz && \
#     tar xzf 1.10.0.tar.gz && \
#     rm 1.10.0.tar.gz
# RUN ls /cub-1.10.0
# ENV CUB_HOME=/cub-1.10.0
# RUN pip install iopath fvcore 
# RUN git clone https://github.com/facebookresearch/pytorch3d.git && cd pytorch3d && git checkout v0.7.7 && FORCE_CUDA=1 pip install -e .
# Install pytorch3d from pre-build binaries, because I couldn't manage to build from source: https://github.com/facebookresearch/pytorch3d/discussions/1752
RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.7+pt2.0.1cu118

# Lightplane
RUN pip install cogapp triton plotly
RUN git clone https://github.com/OliEfr/lightplane.git && cd lightplane && pip install -e .

# DensePoseCSE
RUN pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

# Others
RUN pip install pandas sqlalchemy plotly hydra-core tensorboard lpips opencv-python imageio[ffmpeg] numpy==1.24.3


WORKDIR /home