FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System packages. This repo's own low-level Anomagic.generate()/training-step path (the one
# verified this session, see the handover doc) needs no CUDA-extension JIT compile step (unlike
# AnomalyDiffusion's StyleGAN2/psp_encoder), so this list is a plain Python/opencv runtime stack.
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        locales tzdata \
        curl wget gnupg2 lsb-release sudo software-properties-common gdb \
        python3 python3-dev python3-tk python3-apt python3-venv python3-pip \
        ca-certificates lcov ssh git jq unzip \
        libgl1-mesa-dev libglib2.0-0 libsm6 libxext6 libxrender1 && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 100 && \
    rm -rf /var/lib/apt/lists/*

ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN if [ "$(id -u $USERNAME)" != "$USER_UID" ] || [ "$(id -g $USERNAME)" != "$USER_GID" ]; then \
        groupmod --gid $USER_GID "$(id -gn $USERNAME)" && \
        usermod --uid $USER_UID --gid $USER_GID $USERNAME; \
    fi

# The venv lives outside the workspace mount point (same reasoning as sample/anomalydiffusion's
# devcontainer: the workspace gets bind-mounted over /workspaces/... at container start, which
# would otherwise shadow anything built at that path).
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"
ENV CUDA_HOME=/usr/local/cuda

RUN python3 -m venv $VENV_PATH && \
    chown -R $USERNAME:$USERNAME $VENV_PATH
RUN pip install --no-cache-dir --upgrade pip

# --- Anomagic dependency stack (verified working combination, see handover doc) ---
# torch==2.3.0+cu121 matches env.yml's own pip section (which bundles a CUDA-12.1 runtime via
# nvidia-*-cu12 wheels despite the README's stated "CUDA 11.6+") -- this image's CUDA 12.1
# runtime is the matching base, unlike AnomalyDiffusion's older cu113/11.8 pairing.
RUN pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cu121

# huggingface_hub==0.23.2 is load-bearing: diffusers==0.22.1 without this exact pin resolves
# huggingface_hub to a current release and crashes with
# `ImportError: cannot import name 'cached_download' from 'huggingface_hub'` before any
# Anomagic-specific code runs -- verified this session, not mentioned anywhere in the README.
COPY constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
        diffusers==0.22.1 \
        transformers==4.46.3 \
        peft==0.13.2 \
        accelerate \
        einops \
        safetensors \
        opencv-python \
        "huggingface_hub[cli]==0.23.2"

RUN chown -R $USERNAME:$USERNAME $VENV_PATH

USER $USERNAME

# install claude
RUN curl -fsSL https://claude.ai/install.sh | bash
