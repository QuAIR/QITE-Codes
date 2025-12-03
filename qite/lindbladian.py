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

import os
import sys
from math import ceil
from typing import List, Tuple, Union

import numpy as np
import quairkit as qkit
import torch
from quairkit import Hamiltonian, State, to_state
from quairkit.database import *
from quairkit.qinfo import *

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from qsp import qpp_cir

from .hamiltonian import afm_heisenberg, tf_ising
from .qite import get_qpp_angle
from .utils import even_floor, set_gpu

__all__ = ["to_liouville", "afm_random_jump", "model_ding2024simulating",
           "model_peng2025quantum", "model_yu2025lindbladian", "model_huang2025robust", "algorithm2"]


def to_liouville(system_hamiltonian: Union[Hamiltonian, torch.Tensor], jump_op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Convert the system Hamiltonian and jump operators to the Liouville-space form.
    
    Args:
        system_hamiltonian: The system Hamiltonian.
        jump_op: a batch of jump operators.
        
    Returns:
        A tuple containing two Hamiltonians for the coherent and dissipative parts in Liouville space, where
        jump operators are normalized such that the largest eigenvalue of the dissipative part is 1.
    
    """
    if isinstance(system_hamiltonian, Hamiltonian):
        system_hamiltonian = system_hamiltonian.matrix
    dim = system_hamiltonian.shape[-1]

    assert jump_op.ndim == 3 and list(jump_op.shape)[1:] == [dim, dim], \
        f"Jump operators shape does not match: expect (N, {dim}, {dim}), received {jump_op.shape}"
    _eye = eye(dim)

  
    H = (nkron(_eye, dagger(jump_op) @ jump_op) + 
         nkron(jump_op.mT @ jump_op.conj(), _eye) -
         nkron(jump_op.conj(), jump_op) - 
         nkron(jump_op.mT, dagger(jump_op))).sum(dim=0)
    H /= 2

    normalize_coef = max(torch.linalg.eigvalsh(H).abs().max().item(), 1.0)
    H /= normalize_coef

    Hc = nkron(_eye, system_hamiltonian) - nkron(system_hamiltonian.T, _eye)
    Hc += 0.5j * (nkron(jump_op.conj(), jump_op) - nkron(jump_op.mT, dagger(jump_op))).sum(dim=0) / normalize_coef

    return Hc, H


def afm_random_jump(num_qubits: int) -> Tuple[Hamiltonian, torch.Tensor]:
    r"""Generate a system model of antiferromagnetic Heisenberg Hamiltonian with random jump operators.
    The number of jump operators is set to be equal to the number of qubits.
    
    Args:
        num_qubits: Number of qubits in the system.
    """
    num_jump_operators = num_qubits
    jump_operators = torch.stack([random_hermitian(num_qubits) - 1j * random_hermitian(num_qubits) for _ in range(num_jump_operators)])
    return afm_heisenberg(num_qubits), jump_operators


def _expand_matrix(mat: torch.Tensor, num_qubits: int, target_qubit: int) -> torch.Tensor:
    r"""Expand a single-qubit matrix to a multi-qubit system acting on the target qubit.
    
    Args:
        mat: single-qubit matrix.
        num_qubits: number of qubits in the system.
        target_qubit: the qubit index (0-indexed) on which the matrix acts.
    """
    _eye = eye(2)
    ops = [_eye for _ in range(num_qubits)]
    ops[target_qubit] = mat
    return nkron(*ops)

def model_ding2024simulating(num_qubits: int) -> Tuple[Hamiltonian, torch.Tensor]:
    r"""Generate a system model demonstrated in Ding et al., Simulating Open Quantum Systems Using Hamiltonian Simulations.
    
    Args:
        num_qubits: Number of qubits in the system.
    
    Note:
        Refer to Equation 45 of the paper.
    """
    jump_operators = torch.stack([np.sqrt(0.1) / 2 * 
                                  (_expand_matrix(x(), num_qubits, j) - 
                                   1j * _expand_matrix(y(), num_qubits, j)) 
                                  for j in range(num_qubits)])
    return tf_ising(num_qubits, J=-1, h=-1), jump_operators


def model_peng2025quantum(num_qubits: int) -> Tuple[Hamiltonian, torch.Tensor]:
    r"""Generate a system model demonstrated in Peng et al., Quantum-Trajectory-Inspired Lindbladian Simulation.
    
    Args:
        num_qubits: Number of qubits in the system.
        
    Note:
        Refer to Equation B1 and B4 of the paper for 'tfim' and 'dispXY', respectively.
    """
    _eye, _x, _y, _z = eye(2), x(), y(), z()
    jump_operators = torch.stack([nkron(*[_x for _ in range(num_qubits)]),
                                  nkron(*[_y for _ in range(num_qubits)]),
                                  nkron(*[_z for _ in range(num_qubits)]),
                                  nkron(*[_eye for _ in range(num_qubits)])])
    return tf_ising(num_qubits, J=1, h=-0.5), jump_operators


def model_yu2025lindbladian(num_qubits: int) -> Tuple[Hamiltonian, torch.Tensor]:
    r"""Generate a system model demonstrated in Yu et al., Lindbladian Simulation with Logarithmic Precision Scaling via Two Ancillas.
    
    Args:
        num_qubits: Number of qubits in the system.
        
    Note:
        Refer to Equation 82 of the supplementary material.
    """
    op = np.sqrt(1.5) * nkron(zero_state(1).ket @ one_state(1).bra, eye(2**(num_qubits - 1)))
    return tf_ising(num_qubits, J=-0.1, h=0.2), op.unsqueeze(0)


def model_huang2025robust(num_qubits: int) -> Tuple[Hamiltonian, torch.Tensor]:
    r"""Generate a system model demonstrated in Huang et al., Towards Robust Variational Quantum Simulation of Lindblad Dynamics via Stochastic Magnus Expansion.
    
    Args:
        num_qubits: Number of qubits in the system.
        
    Note:
        Refer to Equation 24 of the paper.
    """
    jump_operators = torch.stack([np.sqrt(0.1) * 
                                  (_expand_matrix(x(), num_qubits, j) + 
                                   1j * _expand_matrix(y(), num_qubits, j)) 
                                  for j in range(num_qubits)])
    return tf_ising(num_qubits, J=1, h=-0.5), jump_operators


def _resource_cost(step_cost: int, list_prob: List[float]) -> int:
    r"""Calculate the resource cost given the step cost and success probabilities.
    
    Args:
        step_cost: the cost per algorithm step.
        list_prob: a list of success probabilities for each step.
    
    """
    list_step_cost = []
    list_step_cost.extend(
        np.ceil(1 / np.prod(list_prob[k:]))
        for k in range(len(list_prob))
    )
    return int(step_cost * sum(list_step_cost))


def algorithm2(t: float, num_step: int, rho0: State, Hc: torch.Tensor, H: torch.Tensor) -> Tuple[float, int]:
    r"""Performs the Lindbladian simulation using ITE algorithm (Algorithm 2 in the paper).
    
    Args:
        t: total evolution time.
        num_step: number of algorithm steps.
        rho0: initial state in density matrix form.
        Hc: The coherent part Hamiltonian in Liouville space.
        H: The dissipative part Hamiltonian in Liouville space.
        alpha: coefficient for QSP angle calculation.
        
    Returns:
        A tuple containing the infidelity and resource cost of the simulation.
        
    Note:
        The resource cost is defined as the expected number of total applications of Hc and H's evolution operators.
    
    """
    ground_state_energy = torch.linalg.eigvalsh(H).min().item()
    if np.abs(ground_state_energy) < 1e-8:
    # if the model of jump operators is trivial, i.e., Hermitian, one can trivially set to 0
        guess_lambda = 0
    else:
        guess_lambda = np.abs(ground_state_energy) + 1 / t
    
    num_qubits = rho0.num_qubits
    unnormalized_rho0 = rho0.vec
    initial_state = to_state(unnormalized_rho0 / torch.norm(unnormalized_rho0), system_dim=2 ** (2 * num_qubits))
    
    ideal_lindbladian = torch.matrix_exp(-1j * (Hc - 1j * H) * t)
    unnormalized_rhot = ideal_lindbladian @ unnormalized_rho0
    expect_state = to_state(unnormalized_rhot / torch.norm(unnormalized_rhot), system_dim=2 ** (2 * num_qubits))
    
    tau = t / num_step
    deg = 586
    
    UHc = torch.linalg.matrix_exp(-1j * Hc * tau)
    UH = torch.linalg.matrix_exp(-1j * H)
    
    list_theta, list_phi = get_qpp_angle(guess_lambda=guess_lambda, tau=tau, deg=deg, alpha=np.exp(-1/num_step))
    
    set_gpu()
    
    device = qkit.get_device()
    initial_state = initial_state.to(device=device)
    expect_state = expect_state.to(device=device)
    
    cir = qpp_cir(list_theta, list_phi, UH.to(device))
    cir.oracle(UHc.to(device), 1)
        
    psi = nkron(zero_state(1), initial_state).ket
    
    list_prob = []

    projector = nkron(zero_state(1).density_matrix, eye(UH.shape[-1]))
    op = projector @ cir.matrix
    for _ in range(num_step):
        output_state = op @ psi
        output_state_norm = torch.norm(output_state).squeeze()
        
        psi = output_state / output_state_norm
        success_prob = output_state_norm ** 2
        list_prob.append(success_prob.item())
            
    infidelity = np.abs(1 - torch.abs(dagger(psi) @ nkron(zero_state(1), expect_state).ket).item())
    qkit.set_device("cpu")
    
    per_step_cost = 2 * deg
    resource_cost = _resource_cost(per_step_cost, list_prob)
    return infidelity, resource_cost
