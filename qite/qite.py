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
from typing import Callable, List, Tuple, Union

import numpy as np
import quairkit as qkit
import torch
from quairkit import Hamiltonian, State
from quairkit.database import zero_state
from quairkit.qinfo import nkron

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from qsp import *

from .sampling import algorithm4

__all__ = ["E_eval", "get_qpp_angle"]


def line_mapping(
    start: Tuple[int, int], end: Tuple[int, int]
) -> Callable[[np.ndarray], np.ndarray]:
    r"""Create a linear mapping function from start to end."""
    x1, y1 = start
    x2, y2 = end
    if x1 >= x2:
        raise ValueError(f"x1 must be less than x2, but got {x1} > {x2}")

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    def map_to_line(x_values: np.ndarray) -> np.ndarray:
        return m * x_values + b

    return map_to_line


def target_func_Q(
    F: Callable[[np.ndarray], np.ndarray], deg: int
) -> Callable[[np.ndarray], np.ndarray]:
    r"""
    Generates and returns a new function based on the input function F and degree modifier.

    Args:
        F: A callable that takes an numpy ndarray and returns an ndarray representing the original function.
        deg: A degree modifier used in computing the phase part of the function.

    """

    def f(x: np.ndarray) -> np.ndarray:
        amplitude = np.sqrt(1 - (F(x) ** 2))
        phase = np.exp(1j * x * deg / 2 * 0.99)
        return amplitude * phase

    return f


def get_qpp_angle(
    guess_lambda: float, tau: int, deg: int, learn: bool = False
) -> Union[Laurent, Tuple[List[float], List[float]]]:
    r"""Generate the QPP angles for the given function.

    Args:
        guess_lambda: lambda parameter for the function.
        tau: time parameter for the function.
        deg: degree of the polynomial approximation.

    """
    lambda_name = round(guess_lambda, 8)
    try:
        file = f"data/lam{lambda_name}_tau{tau}_deg{deg}"
        str_theta = f"{file}_theta.npy"
        str_phi = f"{file}_phi.npy"

        list_theta = np.load(file=str_theta)
        list_phi = np.load(file=str_phi)
        return list_theta, list_phi

    except Exception:
        start, end = -1, -1 + np.pi

        def f(x):
            alpha = 0.85
            return np.where(
                x <= guess_lambda,
                np.exp(tau * (x - guess_lambda)) * alpha,
                -1
                * np.exp(tau * alpha * (-x + guess_lambda) / (1 - alpha))
                * (1 - alpha)
                + 1,
            )

        left_y, right_y = f(start), f(end)

        def f_left(x):
            return left_y + right_y - f((x + 1) % np.pi - 1)

        def f_right(x):
            return left_y + right_y - f((x + 1) % np.pi - 1)

        def target_func(x):
            result = np.zeros_like(x, dtype=float)

            center = (start <= x) & (x <= end)
            left = x < start
            right = x > end

            result[center] = f(x[center])
            result[left] = f_left(x[left])
            result[right] = f_right(x[right])
            return result

        F = to_smooth(target_func, [start, end], transition_width=0.1)
        P = laurent_generator(F, deg)
        P = P * (0.999999 / P.max_norm)

        F_Q = target_func_Q(P, deg=deg)
        Q = laurent_generator(F_Q, deg)
        Q = train_Q(deg=deg, P=P, Q=Q, max_epochs=5000)

        list_theta, list_phi = qpp_angle_approximator(P, Q)

        if learn:
            qkit.set_device("cuda" if torch.cuda.is_available() else "cpu")

            def weight_func(x):
                return 1 + np.exp(-np.log(tau) * np.abs(guess_lambda - x))

            list_theta, list_phi = qpp_angle_learner(
                target_func,
                list_theta,
                list_phi,
                num_sample=1000,
                start=-1,
                end=1 + 0.2,
                key_points=[guess_lambda],
                is_real=True,
                weight=weight_func,
            )
            qkit.set_device("cpu")

        np.save(f"data/lam{lambda_name}_tau{tau}_deg{deg}_theta.npy", np.array(list_theta).astype(np.float64))
        np.save(f"data/lam{lambda_name}_tau{tau}_deg{deg}_phi.npy", np.array(list_phi).astype(np.float64))
        return list_theta, list_phi


def E_eval(
    H: Hamiltonian,
    guess_lambda: float,
    tau: int,
    deg: int,
    phi: State,
    B: float = 1,
    learn: bool = False,
) -> float:
    r"""Evaluate the loss function E.

    Args:
        H: Hamiltonian operator.
        guess_lambda: lambda parameter for the function.
        tau: time parameter for the function.
        deg: degree of the polynomial approximation.
        phi: initial state.

    """
    lambda_name = round(guess_lambda, 8)
    try:
        file = f"data/lam{lambda_name}_tau{tau}_deg{deg}"
        str_theta = f"{file}_theta.npy"
        str_phi = f"{file}_phi.npy"

        list_theta = np.load(file=str_theta)
        list_phi = np.load(file=str_phi)

    except Exception:
        list_theta, list_phi = get_qpp_angle(guess_lambda, tau, deg, learn=learn)

    num_qubits = H.n_qubits
    H_matrix = H.matrix
    input_state = nkron(zero_state(1), phi)

    U = torch.matrix_exp(-1j * H_matrix)
    cir = qpp_cir(list_theta, list_phi, U)
    output_state = cir(input_state)
    del cir

    E, sample_plus_1, sample_minus_1, shots_b0_0 = algorithm4(
        output_state=output_state, hamiltonian=H, num_qubits=num_qubits, tau=tau, B=B
    )
    return E, sample_plus_1, sample_minus_1, shots_b0_0
