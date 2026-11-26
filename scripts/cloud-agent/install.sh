#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Cloud Agent install — TensorRT-LLM checkout bootstrap (idempotent).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DEBIAN_FRONTEND=noninteractive
APT_PACKAGES=(
  python3.12-venv
  python3.12-dev
  git-lfs
  libopenmpi-dev
  libnvidia-compute-580
)
for pkg in "${APT_PACKAGES[@]}"; do
  dpkg -s "$pkg" >/dev/null 2>&1 || sudo apt-get install -y -qq "$pkg"
done

git lfs install --skip-repo 2>/dev/null || git lfs install

VENV_DIR="${REPO_ROOT}/.venv-dev"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -U 'pip>=24' setuptools wheel

# Runtime wheel (CPU agents without GPUs): prefer latest published rc on pypi.nvidia.com.
# Editable installs need a matching precompiled wheel or a full C++/Docker build.
WHEEL_VER="${TRTLLM_WHEEL_VERSION:-1.4.0rc4}"
if ! python -c "import tensorrt_llm" 2>/dev/null; then
  python -m pip install "tensorrt_llm==${WHEEL_VER}" \
    --extra-index-url https://pypi.nvidia.com
fi

# CONTRIBUTING.md minimal hook stack (full requirements-dev.txt re-resolves torch/CUDA wheels).
python -m pip install \
  pre-commit pyrefly==1.4.0 'tomli>=2; python_version < "3.11"' \
  ruff==0.9.4 pytest pytest-asyncio pytest-env pytest-mock pytest-xdist \
  pytest-timeout==2.4.0 pytest-split parameterized

if [[ -z "$(git config --get core.hooksPath 2>/dev/null || true)" ]]; then
  pre-commit install --install-hooks
fi

# Optional Cloud Agent secret (configure in Environment → Secrets).
# Used for NVCR/OCI auth and NVIDIA cloud APIs (e.g. integrate.api.nvidia.com).
if [[ -n "${NVIDIA_API_KEY:-}" ]]; then
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin 2>/dev/null || true
  fi
fi
