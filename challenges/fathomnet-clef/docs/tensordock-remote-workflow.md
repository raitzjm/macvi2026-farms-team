# TensorDock Remote Workflow Policy

This document defines the standard TensorDock workflow for FathomNet-CLEF.

## Purpose

For this challenge, a persistent TensorDock VM controlled remotely from a local agent session is a first-class workflow.

- Preferred pattern: the agent runs locally and uses SSH to operate the TensorDock machine
- Typical remote actions: start or cancel training, inspect `nvidia-smi`, read logs, edit files, manage `tmux` sessions, and run Kaggle submissions
- Preferred environment model on TensorDock: install host-level tooling such as Miniforge, Kaggle CLI, `tmux`, and `git-lfs`, then run the challenge bootstrap scripts directly on the VM host
- The repo container workflow is optional on TensorDock; it is not required when agents are orchestrating the VM remotely over SSH
- TensorDock dataset roots should be exposed via `FATHOMNET_DATA_DIR` rather than assuming the local container path

## Remote Git Sync Rules

- The remote TensorDock clone may be used for normal git work, including commit, pull, and push
- Agents must keep the local workspace and the TensorDock clone in sync explicitly
- Before remote training or remote file edits, verify the TensorDock clone is on the intended commit or branch
- After remote changes are committed and pushed, pull those changes into the local workspace before continuing local work
- After local changes are committed and pushed, pull those changes on TensorDock before continuing remote work
- Do not let local and remote copies drift silently during active experimentation

## Remote Environment Rules

- When using this workflow, the challenge envs are expected to be created on the VM host from the challenge directory after installing Miniforge there
- Keep challenge-specific tooling inside the challenge envs where practical
- Treat the host NVIDIA driver as external infrastructure; validate CUDA access after VM changes or GPU reconfiguration
- Remote training launches that should log to WandB must provide `WANDB_API_KEY` in the actual `ssh` or `tmux` launch environment; do not assume a value exported late in `~/.bashrc` will appear in non-interactive shells
- A prior successful `wandb login` on the VM is useful for verification, but the tracked training scripts still key off `WANDB_API_KEY` being present when deciding whether to enable WandB

## Capacity Monitoring

Agents orchestrating work on TensorDock must monitor disk usage between experiments and before starting large new runs.

- Check free space before starting large training or inference jobs
- Re-check free space between experiments, especially after runs that produce checkpoints, logs, or cached artifacts
- Notify the user early if disk usage is getting close to full so the TensorDock disk size can be increased before jobs fail
- Do not wait for a run to fail with `ENOSPC` before reporting disk pressure

## Session-Specific Access

- Do not hardcode session-specific SSH access details such as usernames, hostnames, IPs, or keys in tracked documentation
- Ask the user for the current SSH user and host when needed for a live session
