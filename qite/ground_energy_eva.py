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
from typing import Callable, Tuple

import numpy as np
import torch
from quairkit import Hamiltonian, State
from quairkit.database import *
from quairkit.qinfo import *

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import math

from qsp import *

from .qite import E_eval
from .sampling import binomial_Proportion_Confidence_Interval

__all__ = ['even_floor', 'algorithm1', 'binary_search']

def even_floor(x):
    num = math.floor(x)
    if num % 2 != 0: # If the number is odd, add 1 to make it even
        num += 1
    return num

def binary_search(tau, H: Hamiltonian, phi: State, deg: int, b_tau: np.ndarray, only_P: bool = False, learn: bool = False) -> np.ndarray:
    r"""
    Performs a binary search-like procedure to find the optimal lambda value such that 
    the estimated energy E(λ - δ; τ) satisfies a threshold condition E < -b_tau.

    Args:
        tau: The parameter controlling precision and delta decay.
        H: The Hamiltonian.
        phi: The input quantum state used in energy evaluation.
        deg: Degree parameter used in qpp calculations inside E_eval.
        b_tau: Threshold value serving as an lower bound for acceptable errors to ensure 
                they do not interfere with the ternary search.
        only_P: If True, only computes part of the energy (e.g., projection P). Defaults to False.
        learn: Enables learning mode which may alter how E_eval behaves. Defaults to False.

    Returns:
        The final adjusted lambda value, serving as the optimized result.
    
    Algorithm Summary:
        - Starts with an initial guess for lambda.
        - Evaluates the energy at lambda and lambda - delta.
        - Adjusts delta in a halving manner (similar to binary search).
        - Stops when either the energy falls below -b_tau or delta becomes too small.
    """
    # Initialize parameters
    i = 0
    delta = 1
    lambda_i = 1 + 1 / tau
    
    # Estimate initial values
    E_lambda_i, _, _, _ = E_eval(H=H, deg=deg, guess_lambda=lambda_i, tau=tau, phi=phi, only_P=only_P, b_tau=b_tau, learn=learn)
    E_lambda_i_minus_delta, _, _, _ = E_eval(H, deg=deg, guess_lambda=lambda_i - delta, tau=tau, phi=phi, only_P=only_P, b_tau=b_tau, learn=learn)
    
    # Check initial conditions
    if E_lambda_i <= - b_tau:
        return lambda_i - 1 / tau
    if E_lambda_i_minus_delta > - b_tau:
        return lambda_i - delta
    
    # Main loop
    while delta >= 2 / tau and E_lambda_i > - b_tau:
        i += 1
        delta = 1 / (2**i)
        
        # Estimate new value
        E_lambda_i_minus_delta, _, _, _ = E_eval(H, deg=deg, guess_lambda=lambda_i - delta, tau=tau, phi=phi, only_P=only_P, b_tau=b_tau, learn=learn)
        
        if E_lambda_i_minus_delta > - b_tau:
            lambda_i -= delta
            
            # Update E_lambda_i for next iteration
            E_lambda_i = E_lambda_i_minus_delta

    
    return lambda_i - 1 / tau


def algorithm1(tau, H: Hamiltonian, phi: State, lambda_0: float, deg: int, only_P: bool = False, b_tau: np.ndarray = None, learn: bool = False, delta_tau: float = 2.5) -> np.ndarray:
    r"""
    Performs a ternary search to find the optimal lambda value such that it lies in [ |lambda_0|, |lambda_0| + 1/tau ].

    Args:
        tau: The imaginary time.
        H: The Hamiltonian.
        phi: The input quantum state.
        lambda_0: Initial guess for lambda from the binary search.
        deg: Degree parameter used in qpp calculations inside E_eval.
        only_P: If True, only computes Laurent. Defaults to False.
        b_tau: Threshold value serving as an lower bound for acceptable errors. Defaults to None.
        learn: Enables learning mode of qpp angle calculation. Defaults to False.

    Returns:
        The final adjusted lambda value after optimization.

    """
    pauli_sum = np.abs(H.coefficients).sum()

    # Initialize parameters
    i = 0
    delta = lambda_0 / 3
    lambda_l = 0
    lambda_r = lambda_0

    E_tau = 1
    error_tau = 0

    List_E_tau = []
    List_tau = []

    sample_plus_1 = 0
    sample_minus_1 = 0
    shots_b0_0 = 0

    List_error = []
    List_resource = [0]

    while delta >= 1 / (2 * tau) or (len(List_E_tau) > 1 and (abs(List_E_tau[-1] - List_E_tau[-2]) > error_tau)) or (len(List_E_tau) > 2 and (abs(List_E_tau[-2] - List_E_tau[-3]) > List_error[-2])):
        i += 1
        List_tau.append(tau)
        delta = (lambda_r - lambda_l) / 3
        deg = max(50, even_floor(0.056 * (tau**3) -4.47 * (tau**2) + 121.36 * tau - 500))
        
        E_lambda_r_minus_2delta, sample_plus_1_l, sample_minus_1_l, shots_b0_0_l = E_eval(H, deg=deg, guess_lambda=lambda_r - 2 * delta, tau=tau, phi=phi, only_P=only_P, b_tau=b_tau, learn=learn)

        E_lambda_r, sample_plus_1_r, sample_minus_1_r, shots_b0_0_r = E_eval(H, deg=deg, guess_lambda=lambda_r, tau=tau, phi=phi, only_P=only_P, b_tau=b_tau, learn=learn)

        sample_plus_1 = sample_plus_1_r
        sample_minus_1 = sample_minus_1_r
        shots_b0_0 = shots_b0_0_r
        # Calculate r2
        r2 = (E_lambda_r_minus_2delta - E_lambda_r) / E_lambda_r
        
        if abs(r2 - (math.exp(4 * tau * delta) - 1)) > tau**(-1) * (math.exp(4 * tau * delta) + 1):
            lambda_l = lambda_r - 2 * delta

            tau += delta_tau
            E_tau, error_tau = binomial_Proportion_Confidence_Interval(sample_plus_1, sample_minus_1, pauli_sum=pauli_sum)
            List_E_tau.append(E_tau)
            List_error.append(error_tau)
            List_resource.append(List_resource[-1] + deg)
            continue
        
        # Update lambda_r and lambda_l
        lambda_r = lambda_r - delta

        tau += delta_tau
        sample_plus_1 += sample_plus_1_l
        sample_minus_1 += sample_minus_1_l
        shots_b0_0 += shots_b0_0_l
        E_tau, error_tau = binomial_Proportion_Confidence_Interval(sample_plus_1, sample_minus_1, pauli_sum=pauli_sum)
        List_E_tau.append(E_tau)
        List_error.append(error_tau)
        List_resource.append(List_resource[-1] + deg)

    List_resource = List_resource[1:]
    return List_tau, List_E_tau, List_error, List_resource
