FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Optional: for users in China mainland
RUN set -eux; \
    cp /etc/apt/sources.list /etc/apt/sources.list.bak; \
    . /etc/os-release; echo "Using UBUNTU_CODENAME=$UBUNTU_CODENAME"; \
    sed -i "s|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g" /etc/apt/sources.list; \
    sed -i "s|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g" /etc/apt/sources.list; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        git \
        wget \
        curl \
        ca-certificates \
        vim \
        tmux \
        unzip \
        sox \
        libsox-fmt-all \
        libsox-fmt-mp3 \
        libsndfile1-dev \
        ffmpeg \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN pip install \
    packaging \
    editdistance \
    gpustat \
    wandb \
    einops \
    tqdm \
    soundfile \
    matplotlib \
    scipy \
    sentencepiece \
    pandas \
    h5py

RUN pip install \
    hydra-core \
    omegaconf \
    deepspeed \
    kaldiio \
    peft==0.6.0 \
    transformers==4.46.3 \
    torchaudio==2.4.0 \
    funasr \
    modelscope \
    accelerate

RUN pip install openai-whisper flashlight-text



