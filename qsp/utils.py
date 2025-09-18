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

from math import log2
from typing import Callable, Optional, Union

import numpy as np
import torch
from quairkit import get_dtype, get_float_dtype
from quairkit.circuit import Circuit
from quairkit.qinfo import dagger

from .angles import qpp_angle_approximator
from .laurent import laurent_generator, pair_generation, revise_tol

r"""
QPP circuit and related tools, see Theorem 6 in paper https://arxiv.org/abs/2209.14278 for more details.
"""


__all__ = ["qpp_cir", "simulation_cir"]


def qpp_cir(
    list_theta: Union[np.ndarray, torch.Tensor],
    list_phi: Union[np.ndarray, torch.Tensor],
    U: Union[np.ndarray, torch.Tensor, float],
) -> Circuit:
    r"""Construct a quantum phase processor of QPP by `list_theta` and `list_phi`.

    Args:
        list_theta: angles for :math:`R_Y` gates.
        list_phi: angles for :math:`R_Z` gates.
        U: unitary or scalar input.

    Returns:
        a multi-qubit version of trigonometric QSP.

    """
    complex_dtype = get_dtype()
    float_dtype = get_float_dtype()

    if not isinstance(list_theta, torch.Tensor):
        list_theta = torch.tensor(list_theta.astype('float64') if isinstance(list_theta, np.ndarray) else list_theta, dtype=float_dtype)
    if not isinstance(list_phi, torch.Tensor):
        list_phi = torch.tensor(list_phi.astype('float64') if isinstance(list_phi, np.ndarray) else list_phi, dtype=float_dtype)
    if not isinstance(U, torch.Tensor):
        U = torch.tensor(U, dtype=complex_dtype)
    if len(U.shape) == 1:
        U = U.to(float_dtype)
    list_theta, list_phi = np.squeeze(list_theta), np.squeeze(list_phi)

    assert len(list_theta) == len(list_phi)
    L = len(list_theta) - 1

    cir = Circuit(2, [2, U.shape[-1]]) if len(U.shape) > 1 else Circuit(1)
    for i in range(L):
        cir.rz(0, param=list_phi[i])
        cir.ry(0, param=list_theta[i])

        # input the unitary
        if len(U.shape) == 1:
            cir.rz(0, param=U)
        elif i % 2 == 0:
            cir.oracle(U, [0, 1], latex_name=r"U", control_idx=1)
        else:
            cir.oracle(dagger(U), [0, 1], latex_name=r"U^\dagger", control_idx=0)
    cir.rz(0, param=list_phi[-1])
    cir.ry(0, param=list_theta[-1])

    return cir

def simulation_cir(
    fn: Callable[[np.ndarray], np.ndarray],
    U: Union[np.ndarray, torch.Tensor, float],
    deg: Optional[int] = 50,
    length: Optional[float] = np.pi,
    step_size: Optional[float] = 0.00001 * np.pi,
    tol: Optional[float] = 1e-30,
) -> Circuit:
    r"""Return a QPP circuit approximating `fn`.

    Args:
        fn: function to be approximated.
        U: unitary input.
        deg: degree of approximation, defaults to be :math:`50`.
        length: half of approximation width, defaults to be :math:`\pi`.
        step_size: sampling frequency of data points, defaults to be :math:`0.00001 \pi`.
        tol: error tolerance, defaults to be :math:`10^{-30}`.

    Returns:
        a QPP circuit approximating `fn` in Theorem 6 of paper https://arxiv.org/abs/2209.14278.

    """
    f = laurent_generator(fn, step_size, deg, length)
    if np.abs(f.max_norm - 1) < 1e-2:
        f = f * (0.999999999 / f.max_norm)
    P, Q = pair_generation(f)
    revise_tol(tol)
    list_theta, list_phi = qpp_angle_approximator(P, Q)

    return qpp_cir(list_theta, list_phi, U)
