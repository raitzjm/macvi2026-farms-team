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
    python3-venv 


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

# ── mmseg dependency stack (Option C: Pixi) ─────────────────────────────────
# System libs required by opencv-python (PyPI wheel links against these)
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        libxcb1 libxcb-xinerama0 libgl1 libglib2.0-0 \
    && sudo rm -rf /var/lib/apt/lists/*

# Install Pixi package manager
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/home/farms/.pixi/bin:$PATH"

# Copy the workspace pixi manifest to a fixed system location for Docker baking.
# We install the "base" environment here (no editable mmseg install — that is
# done at runtime once the workspace is mounted; see challenge CLAUDE.md).
COPY --chown=farms:farms \
    challenges/embedded-semantic-segmentation/pixi.toml \
    /opt/macvi-env/pixi.toml

# Pre-install the base conda+PyPI environment:
#   Python 3.11, PyTorch 2.1.2+cu121, mmcv-full 1.7.2 (prebuilt wheel), numpy<2
# UV_HTTP_TIMEOUT=300 is required: mmcv-full is ~500 MB and the default 30 s times out.
RUN UV_HTTP_TIMEOUT=300 pixi install -e base \
        --manifest-path /opt/macvi-env/pixi.toml

# Apply mmcv scatter patch: PyTorch 2.1.x passes GPU ids as plain int to
# _get_stream(), but newer PyTorch requires torch.device. One-line fix:
RUN MMCV_FUNCS=$(/opt/macvi-env/.pixi/envs/base/bin/python -c \
        "import mmcv, os; print(os.path.join(os.path.dirname(mmcv.__file__), 'parallel', '_functions.py'))") && \
    sed -i \
        's/streams = \[_get_stream(device) for device in target_gpus\]/streams = [_get_stream(torch.device('\''cuda'\'', device) if isinstance(device, int) else device) for device in target_gpus]/' \
        "$MMCV_FUNCS"

# Put the pre-baked pixi "base" env on PATH so it is always active.
# When the workspace is mounted, run `pip install -e mmsegmentation-macvi/`
# to add the editable mmseg install (see challenge CLAUDE.md for instructions).
ENV PATH="/opt/macvi-env/.pixi/envs/base/bin:$PATH"
# ─────────────────────────────────────────────────────────────────────────────

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