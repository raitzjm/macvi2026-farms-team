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
    # Required by opencv-python (pulled in by mmsegmentation-macvi)
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


# =============================================================================
# ML DEPENDENCY STACK
# Option B: Miniforge (conda) + Python 3.11 + PyTorch 2.1.2 + mmcv-full 1.7.2
#
# To swap the dependency approach in the future, replace everything in this
# section (between the ==== markers) and update the PATH line at the bottom.
# =============================================================================

# --- Miniforge ---
# Installs conda (conda-forge channel) into /home/farms/miniforge3.
# Provides the Python version and base package management.
RUN curl -fsSL \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /home/farms/miniforge3 && \
    rm /tmp/miniforge.sh

# --- Conda env ---
# All ML packages live in the "macvi" env, isolated from the system Python.
RUN /home/farms/miniforge3/bin/conda create -n macvi python=3.11 -y

# --- PyTorch 2.1.2 + CUDA 12.1 ---
# Prebuilt wheel from the official PyTorch index; matches the container's CUDA version.
RUN /home/farms/miniforge3/envs/macvi/bin/pip install \
        torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121

# --- mmcv-full 1.7.2 ---
# mmsegmentation 0.30.0 requires mmcv 1.x (asserted in mmseg/__init__.py).
# A prebuilt wheel exists for cu121+torch2.1, so no source compile is needed.
# Note: mmcv 2.x wheels exist for newer torch but are incompatible with this fork.
RUN /home/farms/miniforge3/envs/macvi/bin/pip install \
        mmcv-full==1.7.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# --- Compatibility pins ---
# setuptools<70: conda-forge 82.x dropped pkg_resources, which mmcv still imports.
# numpy<2: mmcv-full 1.7.2 was compiled against the NumPy 1.x C ABI.
# opencv-python<4.11: 4.11+ declares numpy>=2 as a requirement; pin to a version
#   that works with our numpy<2 constraint (mmcv pulls in opencv as a dep).
RUN /home/farms/miniforge3/envs/macvi/bin/pip install "setuptools<70" "numpy<2" "opencv-python<4.11"

# --- mmcv scatter patch ---
# PyTorch 2.1.x changed _get_stream() to require torch.device instead of int.
# mmcv/parallel/_functions.py still passes a plain int, causing a runtime error
# on any multi-GPU scatter operation. One-line fix applied directly to site-packages.
RUN sed -i \
    's/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device("cuda", device) if isinstance(device, int) else device) for device in target_gpus]/' \
    /home/farms/miniforge3/envs/macvi/lib/python3.11/site-packages/mmcv/parallel/_functions.py

# --- mmsegmentation-macvi ---
# Installed non-editable into site-packages so it is fully baked into the image.
#
# COPY is used rather than --mount=type=bind because Podman (used for builds)
# resolves bind-mount source paths relative to the host CWD at build time, not
# relative to the build context. Relative source paths are therefore unreliable
# across different invocation directories. COPY + delete achieves the same
# outcome (package baked into site-packages, source tree removed) and works
# identically with both Docker and Podman.
#
# At runtime the workspace is mounted at /home/farms/macvi2026-farms-team/, which
# also contains mmsegmentation-macvi/, but Python does NOT import from there.
# A non-editable install has no link back to the source — imports always resolve
# from site-packages. The workspace copy is just for reference.
#
# If mmsegmentation-macvi is updated from upstream, rebuild the image.
COPY challenges/embedded-semantic-segmentation/mmsegmentation-macvi /tmp/mmseg-macvi
# Remove any stale egg-info that may have been left behind by a local pip install
# in the workspace (*.egg-info/ is gitignored but not build-context-ignored, so
# COPY picks it up). A stale egg-info causes pip's PEP 517 get_requires_for_build_wheel
# step to fail when the paths recorded in the egg-info don't match /tmp/mmseg-macvi.
RUN rm -rf /tmp/mmseg-macvi/*.egg-info && \
    /home/farms/miniforge3/envs/macvi/bin/pip install cityscapesscripts && \
    /home/farms/miniforge3/envs/macvi/bin/pip install /tmp/mmseg-macvi && \
    rm -rf /tmp/mmseg-macvi

# Put the macvi env on PATH so it is active by default in every shell session.
ENV PATH="/home/farms/miniforge3/envs/macvi/bin:$PATH"

# =============================================================================
# END ML DEPENDENCY STACK
# =============================================================================


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
