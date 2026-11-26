#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# No long-running services for CPU-focused Cloud Agent development.
# NVIDIA_API_KEY (optional Environment secret) is injected into the agent process env when set.
exit 0
