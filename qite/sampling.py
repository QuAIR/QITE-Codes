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
from collections import defaultdict
from typing import Tuple

import numpy as np
import quairkit as qkit
import torch
from quairkit import Hamiltonian, State, to_state
from quairkit.database import *
from quairkit.qinfo import *
from scipy.stats import norm

__all__ = ["algorithm4", "confidence_interval"]


def construct_sigma(pauli_string: str) -> torch.Tensor:
    r"""Construct the Pauli matrix given its string"""
    pauli_string = pauli_string.upper()
    list_t = []
    for char in pauli_string:
        if char == "X":
            list_t.append(x())
        elif char == "Y":
            list_t.append(y())
        elif char == "Z":
            list_t.append(z())
        elif char == "I":
            list_t.append(eye())
        else:
            raise ValueError(f"Invalid Pauli character: {char}")
    return nkron(*list_t)


def construct_T(pauli_string: str) -> torch.Tensor:
    r"""Construct a similar matrix T

    Args:
        pauli_string: a string representing an n-qubit Pauli \sigma

    Returns:
        T such that T \sigma T^\dagger \equiv Z

    """
    pauli_string = pauli_string.upper()
    list_t = []
    for char in pauli_string:
        if char == "X":
            list_t.append(h())
        elif char == "Y":
            list_t.append(h() @ sdg())
        elif char in ["Z", "I"]:
            list_t.append(eye(2))
        else:
            raise ValueError(f"Invalid Pauli character: {char}")
    return nkron(*list_t)


def safe_prob_sample(prob_distribution, shots, max_chunk_size=1e9):
    r"""
    Safely perform probability sampling, automatically batching large shot numbers

    Args:
        prob_distribution: probability distribution (list or tensor)
        shots: total number of samples
        max_chunk_size: maximum number of shots per batch

    Returns:
        merged sample count dictionary {index: count}

    """
    result = defaultdict(int)

    if shots <= max_chunk_size:
        # No need to split, sample directly
        sample_result = prob_sample(prob_distribution, shots=shots)
        for k, v in sample_result.items():
            result[k] += v
    else:
        # Need to split into chunks
        num_chunks = math.ceil(shots / max_chunk_size)
        chunk_size = math.ceil(shots / num_chunks)  # ensure the total is equal to shots

        for _ in range(num_chunks):
            current_shots = min(chunk_size, shots - sum(result.values()))
            sample_result = prob_sample(prob_distribution, shots=current_shots)
            for k, v in sample_result.items():
                result[k] += v

    return dict(result)


def sample_x(
    output_state,
    hamiltonian: Hamiltonian,
    num_qubits: int,
    pauli_coef: np.ndarray,
    l: int,
    shots: int,
) -> float:
    r"""Step 11-14 in Algorithm 3, where the return value is the sum of all X_i in this for loop"""
    coef = pauli_coef[l]
    pauli_str = hamiltonian.pauli_words[l]
    pauli_sum = np.abs(pauli_coef).sum()

    T, sigma = construct_T(pauli_str), construct_sigma(pauli_str)
    obs = T @ sigma @ dagger(T)

    psi = output_state.evolve(T, list(range(1, num_qubits + 1)))
    prob_distribution = psi.density_matrix.diag().real

    qkit.set_device("cuda" if torch.cuda.is_available() else "cpu")
    device = qkit.get_device()

    prob_distribution = prob_distribution.to(device=device)
    sample_result = safe_prob_sample(prob_distribution, shots=shots)
    # print('sampling time', time.time() - start_time)
    qkit.set_device("cpu")

    sum_x = 0

    coef_sign = int(coef / np.abs(coef))
    num_b0_0 = 0
    count_plus_1 = 0
    count_minus_1 = 0

    for digits, num_sample in sample_result.items():
        num_sample = int(num_sample)
        b0 = int(digits[0])
        b = int(digits[1:], base=2)

        if b0 == 0:
            val = coef_sign * int(obs[b, b])
            if val == 1:
                count_plus_1 += num_sample
            elif val == -1:
                count_minus_1 += num_sample
            else:
                raise ValueError("val is not ±1!")
            num_b0_0 += num_sample

        x_val = (1 - b0) * coef / np.abs(coef) * pauli_sum * int(obs[b, b])
        sum_x += num_sample * x_val
    return sum_x, count_plus_1, count_minus_1, num_b0_0


def algorithm4(
    output_state: State, hamiltonian: Hamiltonian, num_qubits: int, tau: float, B: float
):
    pauli_coef = hamiltonian.coefficients
    
    # Lambda = np.abs(pauli_coef).max()
    # total_shots = min(int(1e9), int(2 * max(Lambda ** 2, 1) * tau ** 3 * (B ** (-2))))
    total_shots = int(1e9)

    pauli_weight = np.abs(pauli_coef) / np.abs(pauli_coef).sum()
    list_shots = np.random.multinomial(total_shots, pauli_weight)

    measure_sample = [
        sample_x(
            output_state=output_state,
            hamiltonian=hamiltonian,
            num_qubits=num_qubits,
            pauli_coef=pauli_coef,
            l=l,
            shots=shots,
        )
        for l, shots in enumerate(list_shots)
    ]
    measure_E_total = sum(x[0] for x in measure_sample)
    sample_plus_1 = sum(x[1] for x in measure_sample)
    sample_minus_1 = sum(x[2] for x in measure_sample)
    shots_b0_0 = sum(x[3] for x in measure_sample)

    measure_E = measure_E_total / total_shots

    return measure_E, sample_plus_1, sample_minus_1, shots_b0_0


def confidence_interval(
    count_plus_1: int, count_minus_1: int, confidence=0.95, pauli_sum=1
) -> Tuple[float, float]:
    r"""
    Compute mean value and error bar from counts of +1 and -1 outcomes.

    Args:
        count_plus_1: Number of times +1 was observed
        count_minus_1: Number of times -1 was observed
        confidence: Confidence level (e.g., 0.95 for 95%)
        pauli_sum: Normalization factor

    Returns:
        mean_value: Estimated expectation value: E[X] = (count_plus_1 - count_minus_1) / total
        error: Symmetric error bar around the mean based on normal approximation
    """

    total = count_plus_1 + count_minus_1
    if total == 0:
        return np.nan, np.nan

    # Estimate probability of +1
    p_hat = count_plus_1 / total

    # Z-score for given confidence level (e.g., 1.96 for 95%)
    z = norm.ppf((1 + confidence) / 2)

    # Standard error of the proportion
    se_p = np.sqrt(p_hat * (1 - p_hat) / total)

    # Map to expectation value and error bar in ±1 range
    mean_value = (count_plus_1 - count_minus_1) / total
    error = 2 * z * se_p  # Scale by 2 to match ±1 range

    # Normalization
    mean_value = mean_value * pauli_sum
    error = error * pauli_sum

    return mean_value, error
