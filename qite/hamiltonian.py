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

import torch
from quairkit import Hamiltonian

__all__ = ["normalized_heisenberg_hamiltonian"]


def heisenberg_hamiltonian(
    n: int = 6,
    hz: float = 1,
    hx: float = 0,
    hy: float = 0,
    hxx: float = 1,
    hyy: float = 1,
    hzz: float = 1,
) -> Hamiltonian:
    pauli_terms = []

    # Single-qubit terms
    for i in range(n):
        if hx != 0:
            pauli_terms.append([hx, f"X{i}"])
        if hy != 0:
            pauli_terms.append([hy, f"Y{i}"])
        if hz != 0:
            pauli_terms.append([hz, f"Z{i}"])

    # Two-qubit terms (OBC)
    for i in range(n - 1):
        if hxx != 0:
            pauli_terms.append([hxx, f"X{i}, X{i + 1}"])
        if hyy != 0:
            pauli_terms.append([hyy, f"Y{i}, Y{i + 1}"])
        if hzz != 0:
            pauli_terms.append([hzz, f"Z{i}, Z{i + 1}"])

    return Hamiltonian(pauli_terms)


def normalized_heisenberg_hamiltonian(
    n: int = 6,
    factor: float = 1,
    hz: float = 1,
    hx: float = 0,
    hy: float = 0,
    hxx: float = 1,
    hyy: float = 1,
    hzz: float = 1,
) -> Hamiltonian:
    r"""Prepare a heisenberg hamiltonian which eigenvalues fall within [-1, 1]

    Args:
        n: number of qubits
        factor: inverse normalized factor, should be larger than or equal to 1

    """
    assert factor >= 1

    H_init = heisenberg_hamiltonian(n, hz, hx, hy, hxx, hyy, hzz)

    max_abs_eigen = (torch.linalg.eigvalsh(H_init.matrix)).abs().max() * factor
    new_pauli_string = [
        [coef / max_abs_eigen, pauli_str] for coef, pauli_str in H_init.pauli_str
    ]
    return Hamiltonian(new_pauli_string)
