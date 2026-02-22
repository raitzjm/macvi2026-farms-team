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