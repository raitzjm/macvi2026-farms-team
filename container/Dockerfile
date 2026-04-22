FROM docker.io/nvidia/cuda:12.6.3-devel-ubuntu24.04



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
    git-lfs \
    python3 \
    python3-pip \
    python3-venv \
    # Required at runtime by opencv-python (installed by per-challenge bootstrap scripts)
    libxcb1 \
    libxcb-xinerama0 \
    libgl1 \
    libglib2.0-0 \
    bash-completion


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

# Configure git system-wide and install gitleaks — both require root.
# Must run before USER switch.
RUN git config --system core.hooksPath .githooks

ARG GITLEAKS_VERSION=8.30.0
RUN curl -fsSL \
        "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" \
        -o /tmp/gitleaks.tar.gz && \
    tar -xzf /tmp/gitleaks.tar.gz -C /tmp gitleaks && \
    mv /tmp/gitleaks /usr/local/bin/gitleaks && \
    chmod +x /usr/local/bin/gitleaks && \
    rm /tmp/gitleaks.tar.gz

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

#Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash

# Node.js (via nvm) — required by Gemini CLI and other npm tools
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash && \
    /bin/bash -c "source $HOME/.nvm/nvm.sh && nvm install --lts"

# Gemini CLI
RUN /bin/bash -c "source $HOME/.nvm/nvm.sh && npm install -g @google/gemini-cli"

# OpenCode — multi-model AI coding agent
RUN curl -fsSL https://opencode.ai/install | bash
ENV PATH="/home/farms/.opencode/bin:$PATH"

# Pyright — Python language server for Claude Code pyright-lsp plugin
# Not available in Ubuntu 24.04 apt repos; installed via npm.
RUN /bin/bash -c "source $HOME/.nvm/nvm.sh && npm install -g pyright"

# --- Miniforge ---
# Installs conda (conda-forge channel) into /home/farms/miniforge3.
# Provides conda/mamba for bootstrap scripts that set up per-challenge envs.
RUN curl -fsSL \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /home/farms/miniforge3 && \
    rm /tmp/miniforge.sh

# Set workspace directory
ENV MACVI_WS="/home/${USERNAME}/macvi2026-farms-team"
WORKDIR ${MACVI_WS}

# Add Claude to PATH
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

# Shell conveniences
RUN echo 'alias claude-skip="claude --dangerously-skip-permissions"' >> /home/farms/.bashrc && \
    echo 'alias opencode-run-skip="opencode --dangerously-skip-permissions"' >> /home/farms/.bashrc

#Set shell profile
ENV TERM="xterm-256color"
ENV COLORTERM="truecolor"

#Restore Debian Frontend ENV
ENV DEBIAN_FRONTEND=""

# Default command
CMD ["/bin/bash"]
