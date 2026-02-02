#!/bin/bash
# Source this when opening a new terminal in Cursor on PACE so conda is available.
# Load module system (needed when terminal uses this as rcfile instead of .bashrc)
if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
fi
# Load conda and optional PyTorch
module load anaconda3
module load pytorch/25 2>/dev/null || true
# Enable conda in this shell
command -v conda >/dev/null 2>&1 && eval "$(conda shell.bash hook)"
# Activate project env
conda activate /tmp/python-venv/dpsgd-auditbench_venv 2>/dev/null || true
# Preserve normal bash behavior (aliases, etc.)
[ -f ~/.bashrc ] && . ~/.bashrc
