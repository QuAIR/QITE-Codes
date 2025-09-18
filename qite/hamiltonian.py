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

__all__ = ["normalize", "heisenberg", "afm_heisenberg"]


def heisenberg(
    n: int,
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


def afm_heisenberg(n: int) -> Hamiltonian:
    r"""Prepare an antiferromagnetic heisenberg hamiltonian for a n-qubit homogeneous chain
    
    Args:
        n: number of qubits
        
    Note:
        Equation 1 in *Quantum Simulation of Antiferromagnetic Heisenberg Chain with Gate-Defined Quantum Dots*
    
    """
    pauli_terms = []
    for i in range(n - 1):
        pauli_terms.extend(
            (
                [1 / n, f"X{i}, X{i + 1}"],
                [1 / n, f"Y{i}, Y{i + 1}"],
                [1 / n, f"Z{i}, Z{i + 1}"],
            )
        )
    pauli_terms.append([-(n - 1) / n, f"I{n - 1}"])
    return Hamiltonian(pauli_terms)


def normalize(H: Hamiltonian, factor: float) -> Hamiltonian:
    r"""Normalize a Hamiltonian which eigenvalues will be divided by factor

    Args:
        H: a given Hamiltonian
        factor: inverse normalized factor, should be larger than or equal to 1

    """
    assert factor >= 1

    new_pauli_string = [
        [coef / factor, pauli_str] for coef, pauli_str in H.pauli_str
    ]
    return Hamiltonian(new_pauli_string)
