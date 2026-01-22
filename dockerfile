FROM hub.szaic.com/sjtu-base/sjtu_base-pytorch-for-ascend:cann8.0.0-torch2.1.0-py3.10

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    sox \
    libsox-fmt-all \
    libsox-fmt-mp3 \
    libsndfile1-dev \
    ffmpeg \
    ninja-build \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir \
    deepspeed \
    torchvision==0.16.0 \
    transformers==4.46.3 \
    torchaudio==2.1.0 \
    packaging \
    editdistance \
    gpustat \
    wandb \
    tqdm \
    soundfile \
    matplotlib \
    sentencepiece \
    pandas \
    h5py \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    kaldiio \
    peft==0.6.0 \
    funasr \
    modelscope \
    openai-whisper

CMD ["/bin/bash"]