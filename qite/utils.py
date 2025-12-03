# !/usr/bin/env python3
# Copyright (c) 2025 QuAIR team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import quairkit as qkit
import torch

__all__ = ["even_floor", "set_gpu"]


def even_floor(x):
    num = math.floor(x)
    if num % 2 != 0:  # If the number is odd, add 1 to make it even
        num += 1
    return num


def set_gpu(min_free_gb: float = 4.0, max_utilization_pct: int = 50) -> None:
    r"""
    Picks a CUDA device with at least `min_free_gb` free memory and <= `max_utilization_pct` GPU utilization,
    breaking ties by most free memory. Falls back to 'most free memory' if NVML is unavailable.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        return _efficient_pick(min_free_gb, max_utilization_pct)
    except Exception:
        # Fallback: PyTorch-only approach
        best_idx, best_free_gb = None, -1.0
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
            except TypeError:
                torch.cuda.set_device(i)
                free, total = torch.cuda.mem_get_info()
            free_gb = free / (1024**3)
            if free_gb > best_free_gb:
                best_idx, best_free_gb = i, free_gb

        if best_idx is None or best_free_gb < min_free_gb:
            qkit.set_device("cpu")
            return

        qkit.set_device(f"cuda:{best_idx}")


def _efficient_pick(min_free_gb, max_utilization_pct):
    import pynvml as nvml  # pip install nvidia-ml-py3
    nvml.nvmlInit()
    count = nvml.nvmlDeviceGetCount()

    candidates = []
    for idx in range(count):
        h = nvml.nvmlDeviceGetHandleByIndex(idx)
        mem = nvml.nvmlDeviceGetMemoryInfo(h)   # bytes
        util = nvml.nvmlDeviceGetUtilizationRates(h).gpu  # %
        free_gb = mem.free / (1024**3)

        # Optional: number of running compute processes (API name varies by driver)
        procs = 0
        for name in ("nvmlDeviceGetComputeRunningProcesses_v3",
                         "nvmlDeviceGetComputeRunningProcesses_v2",
                         "nvmlDeviceGetComputeRunningProcesses"):
            if fn := getattr(nvml, name, None):
                try:
                    procs = len(fn(h))
                except nvml.NVMLError:
                    procs = 0
                break

        # Keep if it meets thresholds
        if free_gb >= min_free_gb and util <= max_utilization_pct:
            # Sort key: fewest procs, then lowest util, then most free mem
            candidates.append((procs, util, -free_gb, idx))

    if candidates:
        candidates.sort()
        chosen = candidates[0][-1]
    else:
        # Fallback to "most free memory" using NVML data
        freemem = []
        for idx in range(count):
            h = nvml.nvmlDeviceGetHandleByIndex(idx)
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            freemem.append((-mem.free, idx))  # negate to sort descending
        chosen = sorted(freemem)[0][1]

    nvml.nvmlShutdown()
    qkit.set_device(f"cuda:{chosen}")
