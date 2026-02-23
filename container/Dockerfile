FROM docker.io/nvidia/cuda:12.1.1-devel-ubuntu22.04



# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt upgrade -y && \
    apt-get install --no-install-recommends  -y \
    openssh-client \
    unzip \
    curl \
    tar \
    wget \
    ca-certificates \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libxcb1 \
    libxcb-xinerama0 \
    libgl1 \
    libglib2.0-0


# Create non-root user
ARG USERNAME=farms
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if getent passwd ubuntu > /dev/null 2>&1; then \
        userdel -r ubuntu && \
        echo "Deleted existing ubuntu user"; \
    fi && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "Created new user $USERNAME"


# Add sudo support for the non-root user
RUN apt-get update && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER  ${USERNAME}

#Install Github CLI
RUN (type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y

#Install local packages
RUN curl -fsSL https://claude.ai/install.sh | bash

# --- ML dependency stack: conda + Python 3.11 + PyTorch 2.1.2 + mmcv-full 1.7.2 ---

# Miniforge (conda-forge base)
RUN curl -fsSL \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /home/farms/miniforge3 && \
    rm /tmp/miniforge.sh

# macvi conda env: Python 3.11
RUN /home/farms/miniforge3/bin/conda create -n macvi python=3.11 -y

# PyTorch 2.1.2 + CUDA 12.1 (prebuilt wheel)
RUN /home/farms/miniforge3/envs/macvi/bin/pip install \
        torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121

# mmcv-full 1.7.2 prebuilt wheel (cu121 + torch2.1; no source build needed)
RUN /home/farms/miniforge3/envs/macvi/bin/pip install \
        mmcv-full==1.7.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# setuptools<70: conda-forge 82.x dropped pkg_resources (required by mmcv)
# numpy<2: mmcv-full 1.7.2 was compiled against NumPy 1.x ABI
RUN /home/farms/miniforge3/envs/macvi/bin/pip install "setuptools<70" "numpy<2"

# Patch mmcv scatter: PyTorch 2.1.x passes int device ids to _get_stream(),
# which now expects torch.device — one-line fix applied at build time
RUN sed -i \
    's/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device("cuda", device) if isinstance(device, int) else device) for device in target_gpus]/' \
    /home/farms/miniforge3/envs/macvi/lib/python3.11/site-packages/mmcv/parallel/_functions.py

# Copy and install mmsegmentation-macvi (non-editable; upstream code, never modified)
COPY --chown=farms:farms \
    challenges/embedded-semantic-segmentation/mmsegmentation-macvi \
    /opt/macvi-env/mmsegmentation-macvi
RUN /home/farms/miniforge3/envs/macvi/bin/pip install cityscapesscripts && \
    /home/farms/miniforge3/envs/macvi/bin/pip install /opt/macvi-env/mmsegmentation-macvi

# Put macvi env first on PATH (active by default)
ENV PATH="/home/farms/miniforge3/envs/macvi/bin:$PATH"

# Set workspace directory
ENV MACVI_WS="/home/${USERNAME}/macvi2026-farms-team"
WORKDIR ${MACVI_WS}

# Add Claude to PATH
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

#Set shell profile
ENV TERM="xterm-256color"
ENV COLORTERM="truecolor"

#Restore Debian Frontend ENV
ENV DEBIAN_FRONTEND=""

# Default command
CMD ["/bin/bash"]
