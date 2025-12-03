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
from typing import Callable, List, Tuple

import numpy as np
import torch
from quairkit import Hamiltonian, State
from quairkit.database import *
from quairkit.qinfo import *

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import math

from qsp import *

from .qite import E_eval
from .sampling import confidence_interval
from .utils import even_floor

__all__ = ["algorithm1", "algorithm6"]


def algorithm6(
    tau, H: Hamiltonian, phi: State, deg: int, B: float, learn: bool = False
) -> np.ndarray:
    r"""
    Performs a binary search-like procedure to find the optimal lambda value such that
    the estimated energy E(λ - δ; τ) satisfies a threshold condition E < -B.

    Args:
        tau: The parameter controlling precision and delta decay.
        H: The Hamiltonian.
        phi: The input quantum state used in energy evaluation.
        deg: Degree parameter used in qpp calculations inside E_eval.
        B: Threshold value serving as an lower bound for acceptable errors to ensure
                they do not interfere with the ternary search.
        learn: Enables learning mode which may alter how E_eval behaves. Defaults to False.

    Returns:
        The final adjusted lambda value, serving as the optimized result.

    Algorithm Summary:
        - Starts with an initial guess for lambda.
        - Evaluates the energy at lambda and lambda - delta.
        - Adjusts delta in a halving manner (similar to binary search).
        - Stops when either the energy falls below -B or delta becomes too small.
    """
    # Initialize parameters
    i = 0
    delta = 1
    lambda_i = 1 + 1 / tau

    # Estimate initial values
    E_lambda_i, _, _, _ = E_eval(
        H=H, deg=deg, guess_lambda=lambda_i, tau=tau, phi=phi, B=B, learn=learn
    )
    E_lambda_i_minus_delta, _, _, _ = E_eval(
        H, deg=deg, guess_lambda=lambda_i - delta, tau=tau, phi=phi, B=B, learn=learn
    )

    # Check initial conditions
    if E_lambda_i <= -B:
        return lambda_i - 1 / tau
    if E_lambda_i_minus_delta > -B:
        return lambda_i - delta

    # Main loop
    while delta >= 2 / tau and E_lambda_i > -B:
        i += 1
        delta = 1 / (2**i)

        # Estimate new value
        E_lambda_i_minus_delta, _, _, _ = E_eval(
            H,
            deg=deg,
            guess_lambda=lambda_i - delta,
            tau=tau,
            phi=phi,
            B=B,
            learn=learn,
        )

        if E_lambda_i_minus_delta > -B:
            lambda_i -= delta

            # Update E_lambda_i for next iteration
            E_lambda_i = E_lambda_i_minus_delta

    return lambda_i - 1 / tau


def algorithm1(
    tau: float,
    H: Hamiltonian,
    phi: State,
    guess_lambda: float,
    delta_tau: float,
    criteria: Callable[[List[float], List[float]], bool],
    learn: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Performs a ternary search to solve Problem 1.

    Args:
        tau: guess evolution time.
        H: The Hamiltonian.
        phi: The input quantum state.
        guess_lambda: Initial guess for lambda from the binary search.
        delta_tau: Step size for guess time
        criteria: a function tests for convergence
        learn: Enables learning mode of qpp angle calculation. Defaults to False.

    Returns:
        the return object contains the following items:
        - a list of iterated evolution time
        - a list of ground-state energy estimation
        - a list of estimation error
        - a list of accumulated resource cost
    """
    pauli_sum = np.abs(H.coefficients).sum()

    # Initialize parameters
    delta = guess_lambda / 3
    lambda_l = 0
    lambda_r = guess_lambda

    sample_plus_1 = 0
    sample_minus_1 = 0
    shots_b0_0 = 0

    list_E_tau = []
    list_tau = []
    list_error = []
    list_resource = [0]

    while delta >= 1 / (2 * tau) or (not criteria(list_E_tau, list_error)):
        list_tau.append(tau)
        delta = (lambda_r - lambda_l) / 3
        deg = max(
            50, even_floor(0.056 * (tau**3) - 4.47 * (tau**2) + 121.36 * tau - 500)
        )

        E_lambda_r_minus_2delta, sample_plus_1_l, sample_minus_1_l, shots_b0_0_l = (
            E_eval(
                H,
                deg=deg,
                guess_lambda=lambda_r - 2 * delta,
                tau=tau,
                phi=phi,
                B=None,
                learn=learn,
            )
        )

        E_lambda_r, sample_plus_1_r, sample_minus_1_r, shots_b0_0_r = E_eval(
            H, deg=deg, guess_lambda=lambda_r, tau=tau, phi=phi, B=None, learn=learn
        )

        sample_plus_1 = sample_plus_1_r
        sample_minus_1 = sample_minus_1_r
        shots_b0_0 = shots_b0_0_r

        # Calculate r
        r = (E_lambda_r_minus_2delta - E_lambda_r) / E_lambda_r

        if abs(r - (math.exp(4 * tau * delta) - 1)) > tau ** (-1) * (
            math.exp(4 * tau * delta) + 1
        ):
            lambda_l = lambda_r - 2 * delta

            tau += delta_tau
            E_tau, error_tau = confidence_interval(
                sample_plus_1, sample_minus_1, confidence=0.95, pauli_sum=pauli_sum,
            )

            list_E_tau.append(E_tau)
            list_error.append(error_tau)
            list_resource.append(list_resource[-1] + deg)
            continue

        # Update lambda_r and lambda_l
        lambda_r = lambda_r - delta

        tau += delta_tau
        sample_plus_1 += sample_plus_1_l
        sample_minus_1 += sample_minus_1_l
        shots_b0_0 += shots_b0_0_l
        E_tau, error_tau = confidence_interval(
            sample_plus_1, sample_minus_1, confidence=0.95, pauli_sum=pauli_sum
        )

        list_E_tau.append(E_tau)
        list_error.append(error_tau)
        list_resource.append(list_resource[-1] + deg)

    list_resource = list_resource[1:]
    return list_tau, list_E_tau, list_error, list_resource
