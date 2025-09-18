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

import time
import warnings
from copy import copy
from math import atan, cos, sin
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import quairkit as qkit
import torch
import torch.nn as nn
import torch.optim as optim
from quairkit.circuit import Circuit
from quairkit.database import ry, rz
from scipy.integrate import cumulative_trapezoid

from .laurent import Laurent

r"""
QPP angle solver for trigonometric QSP, see Lemma 3 in paper https://arxiv.org/abs/2205.07848 for more details.
"""


__all__ = ['qpp_angle_approximator', 'qpp_angle_learner', 'train_Q']


def qpp_angle_approximator(P: Laurent, Q: Laurent) -> Tuple[List[float], List[float]]:
    r"""Approximate the corresponding set of angles for a Laurent pair `P`, `Q`.
    
    Args:
        P: a Laurent poly.
        Q: a Laurent poly.
        
    Returns:
        contains the following elements:
        - list_theta: angles for :math:`R_Y` gates;
        - list_phi: angles for :math:`R_Z` gates;
        - alpha: global phase for the last term.
    
    Note:
        qpp_angle_approximator assumes that the only source of error is the precision (which is not generally true).

    """
    list_theta = []
    list_phi = []
    
    # backup P for output check
    P_copy = copy(P)
    
    L = P.deg
    while L > 0:
        theta, phi = update_angle([P.coef[-1].astype(np.clongdouble), P.coef[0].astype(np.clongdouble), Q.coef[-1].astype(np.clongdouble), Q.coef[0].astype(np.clongdouble)])
        
        list_theta.append(theta)
        list_phi.append(phi)
        
        P, Q = update_polynomial(P, Q, theta, phi, verify=False)
        
        L -= 1
        P, Q = P.reduced_poly(L), Q.reduced_poly(L)
    
    # decide theta[0], phi[0] and global phase alpha
    p_0, q_0 = P.coef[0].astype(np.clongdouble), Q.coef[0].astype(np.clongdouble)
    alpha, theta, phi = yz_decomposition(np.array([[p_0, -q_0], [np.conj(q_0), np.conj(p_0)]]))
    list_theta.append(theta)
    list_phi.append(phi)
    
    # test outputs, by 5 random data points in [-pi, pi]
    list_x = np.linspace(-np.pi, np.pi, 300, endpoint=True)
    experiment_y = matrix_generation(list_theta, list_phi, list_x, alpha)[:, 0, 0]
    actual_y = P_copy(list_x)
    print(f"Computations of angles for QPP are completed with mean error {np.abs(experiment_y - actual_y).mean()}")
    
    return list_theta, list_phi


def _construct_learnable_circuit(list_theta: List[float], list_phi: List[float], data: torch.Tensor) -> Circuit:
    r"""Construct a learnable circuit, where initial parameters are `list_theta` and `list_phi`.
    """
    L = len(list_theta) - 1
    list_phi, list_theta = list_phi.view([-1, 1, 1, 1]), list_theta.view([-1, 1, 1, 1])
    
    cir = Circuit(1)
    for i in range(L):
        cir.rz(0, param=torch.nn.Parameter(list_phi[i]))
        cir.ry(0, param=torch.nn.Parameter(list_theta[i]))

        cir.rz(0, param=data)
    
    cir.rz(0, param=torch.nn.Parameter(list_phi[-1]))
    cir.ry(0, param=torch.nn.Parameter(list_theta[-1]))
    cir.rz(0)
    
    return cir


def _align_y(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Align `y` with `x` by finding the optimal phase shift.
    """
    overlap = torch.vdot(x, y)
    optimal_phase = overlap.angle()
    return y * torch.exp(-1j * optimal_phase)

def _weighted_linspace(w: Callable[[np.ndarray], np.ndarray], start: int, end: int, num_points: int) -> np.ndarray:
    """
    Generate points distributed according to a given weight function w(x).

    Parameters:
        w: Weight function w(x), must return positive values.
        start: Start of interval.
        end: End of interval.
        num_points: Number of points to generate.

    Returns:
        Array of points distributed according to w(x).
    
    """
    # Dense grid for numerical integration
    x_dense = np.linspace(start, end, 5000)

    # Evaluate weight function on dense grid
    w_values = w(x_dense)

    # Validate that weight function is positive
    if np.any(w_values < 0):
        raise ValueError("Weight function must be non-negative on the interval.")
    if not np.any(w_values > 0):
        raise ValueError("Weight function must have positive values on the interval.")

    # Compute cumulative integral numerically (CDF)
    cdf_values = cumulative_trapezoid(w_values, x_dense, initial=0)

    # Normalize the CDF
    cdf_values /= cdf_values[-1]

    # Uniformly spaced points in [0,1]
    u_uniform = np.linspace(0, 1, num_points)

    return np.interp(u_uniform, cdf_values, x_dense)


def qpp_angle_learner(F: Union[Laurent, Callable[[np.ndarray], np.ndarray]],
                      list_theta: List[float], list_phi: List[float], 
                      num_sample: int = 1000, start: float = -np.pi, end: float = np.pi, key_points: List[float] = None,
                      is_real: bool = False, weight: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Tuple[List[float], List[float]]:
    r"""Optimize the corresponding set of angles for a Laurent poly `P`.
    
    Args:
        F: a Laurent poly or a function (that supports batched input)
        list_theta: initial angles for :math:`R_Y` gates.
        list_phi: initial angles for :math:`R_Z` gates.
        num_sample: number of samples for optimization.
        start: start point of the learn interval.
        end: end point of the learn interval.
        is_real: whether F is real-valued, defaults to be `False`.
        weight: a function that computes the weight for each sample, defaults to be `None`.
        
    Returns:
        contains the following elements:
        - list_theta: improved angles for :math:`R_Y` gates;
        - list_phi: improved angles for :math:`R_Z` gates.
    
    """
    qkit.set_dtype('complex128')
    
    if weight is None:
        weight = np.ones_like
    
    list_x = _weighted_linspace(weight, start, end, num_sample).astype(np.float64)
    list_x = np.concatenate([list_x, np.array(key_points, dtype=list_x.dtype)]) if key_points is not None else list_x
    
    list_y = np.array(F(list_x).real if is_real else F(list_x), dtype=np.complex128)
    list_w = weight(list_x)
    
    list_theta = np.array(list_theta, dtype=np.float64)
    list_phi = np.array(list_phi, dtype=np.float64)
    list_x, list_y, list_theta, list_phi, list_w = map(torch.tensor, [list_x, list_y, list_theta, list_phi, list_w])
    cir = _construct_learnable_circuit(list_theta, list_phi, list_x)
    
    def loss_fcn(circuit: Circuit) -> torch.Tensor:
        actual_output = circuit.matrix[:, 0, 0]
        expected_output = _align_y(list_y, actual_output)
        return ((torch.abs(actual_output - expected_output) ** 2) * list_w).mean()
    
    def test_fcn(circuit: Circuit) -> float:
        actual_output = circuit.matrix[:, 0, 0]
        expected_output = _align_y(list_y, actual_output)
        return torch.abs((actual_output - expected_output)).max().item()
    
    original_diff = test_fcn(cir)
    if np.isnan(original_diff) or np.isinf(original_diff):
        print("Warning: original_diff is NaN or Inf, using default value.")
        original_diff = 1e-3 
    initial_lr = max(min(original_diff / 20, 1e-1), 1e-6)
    
    opt = torch.optim.Adam(lr=initial_lr, params=cir.parameters()) # cir is a Circuit type
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5) # activate scheduler

    NUM_ITR, time_list = 500, []
    for itr in range(NUM_ITR):
        start_time = time.time()
        opt.zero_grad()

        loss = loss_fcn(cir) # compute loss

        loss.backward()
        opt.step()
        scheduler.step(loss) # activate scheduler
        
        lr = scheduler.get_last_lr()[0]
        time_list.append(time.time() - start_time)
        if itr % (NUM_ITR // 5) == 0 or itr == NUM_ITR - 1 or lr < 1e-6:
            loss = loss.item()
            diff = test_fcn(cir)
            # print(f"iter: {str(itr).zfill(len(str(NUM_ITR)))}, " +
            #       f"loss: {loss:.8f}, lr: {lr:.2E}, max diff: {diff:.8f}, " + 
            #       f"avg_time: {np.mean(time_list):.4f}s")
            if lr < 1e-6:
                break
            
    if diff > original_diff:
        warnings.warn(
            f"Training failed: opt max diff {diff} is larger than original one {original_diff}. Return original one instead.", UserWarning)
        return list_theta.cpu().numpy(), list_phi.cpu().numpy()
    
    opt_theta, opt_phi = [], []
    for idx, gate in enumerate(cir.children()):
        param = gate.info['param']
        if param.numel() == 1:
            gate_name, param = gate.info['name'], param.item()
            if gate_name == 'ry':
                opt_theta.append(param)
            elif gate_name == 'rz':
                if idx == len(cir) - 1:
                    opt_alpha = param
                else:
                    opt_phi.append(param)
    print(f"Optimization of angles for QPP is completed with MSE {loss}")
    
    return opt_theta, opt_phi
    
    

# ------------------------------------------------- Split line -------------------------------------------------
r"""
    Belows are support functions for angles computation.
"""


def update_angle(coef: List[complex]) -> Tuple[float, float]:
    r"""Compute angles by `coef` from `P` and `Q`.
    
    Args:
        coef: the first and last terms from `P` and `Q`.
        
    Returns:
        `theta` and `phi`.
    
    """
    # with respect to the first and last terms of P and Q, respectively
    p_d, p_nd, q_d, q_nd = coef[0], coef[1], coef[2], coef[3]

    # r_p_d = mp.fabs(p_d)
    # r_p_nd = mp.fabs(p_nd)
    # r_q_d = mp.fabs(q_d)
    # r_q_nd = mp.fabs(q_nd)
    # angle_p_d = mp.arg(p_d + 0j)
    # angle_p_nd = mp.arg(p_nd + 0j)
    # angle_q_d = mp.arg(q_d + 0j)
    # angle_q_nd = mp.arg(q_nd + 0j)
    r_p_d = np.abs(p_d)
    r_q_d = np.abs(q_d)
    angle_p_d = np.angle(p_d + 0j)
    angle_q_d = np.angle(q_d + 0j)
    r_p_nd = np.abs(p_nd)
    r_q_nd = np.abs(q_nd)
    angle_p_nd = np.angle(p_nd + 0j)
    angle_q_nd = np.angle(q_nd + 0j)

    if r_p_d != 0 and r_q_d != 0:
        # val = -1 * p_d / q_d
        # return atan(np.abs(val)) * 2, np.real(np.log(val / np.abs(val)) / (-1j))
        
        return atan(-1 * r_p_d / r_q_d) * 2, angle_q_d - angle_p_d
    
    elif r_p_d < 1e-25 and r_q_d < 1e-25:
        # val = q_nd / p_nd
        # return atan(np.abs(val)) * 2, np.real(np.log(val / np.abs(val)) / (-1j))
        
        return atan(r_q_nd / r_p_nd) * 2, angle_q_nd - angle_p_nd
    
    elif r_p_d < 1e-25 and r_q_nd < 1e-25:
        return 0, 0
    
    elif r_p_nd < 1e-25 and r_q_d < 1e-25:
        return np.pi, 0
    
    raise ValueError(
        f"Coef error: check these four coef {[p_d, p_nd, q_d, q_nd]}")


def update_polynomial(P: Laurent, Q: Laurent, theta: float, phi: float, verify: Optional[bool] = True) -> Tuple[Laurent, Laurent]:
    r"""Update `P` and `Q` by `theta` and `phi`.
    
    Args:
        P: a Laurent poly.
        Q: a Laurent poly.
        theta: a param.
        phi: a param.
        verify: whether verify the correctness of computation, defaults to be `True`.
    
    Returns:
        updated `P` and `Q`.
        
    """
    
    phi_hf = phi / 2
    theta_hf = theta / 2
    
    X = Laurent([0, 0, 1])
    inv_X = Laurent([1, 0, 0])
    
    new_P = (X * P * np.exp(1j * phi_hf) * cos(theta_hf)) + (X * Q * np.exp(-1j * phi_hf) * sin(theta_hf))
    new_Q = (inv_X * Q * np.exp(-1j * phi_hf) * cos(theta_hf)) - (inv_X * P * np.exp(1j * phi_hf) * sin(theta_hf))
    
    if not verify:
        return new_P, new_Q
    
    condition_test(new_P, new_Q)
    return new_P, new_Q

    
def condition_test(P: Laurent, Q: Laurent) -> None:
    r"""Check whether `P` and `Q` satisfy:
        - deg(`P`) = deg(`Q`);
        - `P` and `Q` have the same parity;
        - :math:`PP^* + QQ^* = 1`.
    
    Args:
        P: a Laurent poly.
        Q: a Laurent poly.
    
    """
    L = P.deg
    
    if L != Q.deg:
        print("The last and first terms of P: ", P.coef[0], P.coef[-1])
        print("The last and first terms of Q: ", Q.coef[0], Q.coef[-1])
        raise ValueError(f"P's degree {L} does not agree with Q's degree {Q.deg}")
    
    if P.parity != Q.parity or P.parity != L % 2:
        print(f"P's degree is {L}")
        raise ValueError(f"P's parity {P.parity} and Q's parity {Q.parity}) should be both {L % 2}")
    
    poly_one = (P * P.conj) + (Q * Q.conj)
    if poly_one != 1:
        print(f"P's degree is {L}")
        print("the last and first terms of PP* + QQ*: ", poly_one.coef[0], poly_one.coef[-1])
        raise ValueError("PP* + QQ* != 1: check your code")
    

def matrix_generation(list_theta: List[float], list_phi: List[float], x: float, alpha: Optional[float] = 0) -> np.ndarray:
    r"""Return the matrix generated by sets of angles.
    
    Args:
        list_theta: angles for :math:`R_Y` gates.
        list_phi: angles for :math:`R_Z` gates.
        x: input of polynomial P
        alpha: global phase
        
    Returns:
        unitary matrix generated by YZZYZ circuit
        
    """
    assert len(list_theta) == len(list_phi)
    L = len(list_theta) - 1
    
    cir = Circuit(1)
    for i in range(L):
        cir.rz(0, param=list_phi[i])
        cir.ry(0, param=list_theta[i])
        cir.rz(0, param=x)  # input x
    cir.rz(0, param=list_phi[-1])
    cir.ry(0, param=list_theta[-1])
    
    return cir.matrix.numpy() * alpha


def yz_decomposition(U: np.ndarray) -> Tuple[complex, float, float]:
    r"""Return the YZ decomposition of U.
    
    Args:
        U: single-qubit unitary.

    Returns:
        `alpha`, `theta`, `phi` st. :math:`U[0, 0] = \alpha R_Y(\theta) R_Z(\phi) [0, 0]`.
    
    """
    U = torch.from_numpy(U.astype(np.complex128))

    param = torch.nn.Parameter(torch.rand(4, dtype=torch.float64))

    def loss_fcn(alpha_angle, omega, theta, phi) -> torch.Tensor:
        phase = torch.exp(1j * alpha_angle)
        return torch.norm(phase * (rz(omega) @ ry(theta) @ rz(phi)) - U)

    NUM_ITR = 500

    opt = torch.optim.Adam(lr=0.1, params=[param]) # cir is a Circuit type
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5)

    for _ in range(NUM_ITR):
        opt.zero_grad()

        loss = loss_fcn(param[0], param[1], param[2], param[3]) # compute loss

        loss.backward()
        opt.step()

        scheduler.step(loss.item())

    alpha_angle, omega, theta, phi = param.detach().numpy()
    alpha = np.exp(1j * alpha_angle) * rz(omega)[0, 0].item()
    return alpha, theta, phi

    

def filter_laurent_coefficients(c_k, deg):
    """
    Filters the Laurent coefficients based on the parity (even or odd) of 'deg'.
    Coefficients corresponding to frequency components with mismatched parity are set to 0.

    Parameters:
        c_k (array-like): Original Laurent coefficient array of shape (2n+1,)
        deg (int): The reference degree used to determine parity

    Returns:
        numpy.ndarray: Filtered Laurent coefficients d_k of shape (2n+1,)
    """
    # Create a copy to avoid modifying the original data
    d_k = c_k.copy()
    
    # Determine the parity of deg
    is_deg_even = (deg % 2 == 0)
    
    # Traverse all coefficients
    n = (len(c_k) - 1) // 2  # Central index
    
    for k in range(-n, n + 1):
        # Index of current coefficient in the array
        index = k + n
        
        # Determine the parity of the current frequency component
        is_k_even = (k % 2 == 0)
        
        # If the parity of k does not match that of deg, set the coefficient to 0
        if is_deg_even != is_k_even:
            d_k[index] = 0
            
    return d_k


def train_Q(deg, P, Q, max_epochs=500) -> Laurent:
    """
    Train the Laurent Q such that |Q(x)|^2 + |P(x)|^2 ≈ 1.

    Args:
        deg (int): Maximum frequency degree (symmetric around 0).
        P_coef (torch.Tensor): Fixed Laurent coefficients for P.
        Q_coef (torch.Tensor): Initial Laurent coefficients for Q (some will be optimized).
        num_points (int): Number of time-domain points to compute (default: same as Q_coef size).
        max_epochs (int): Maximum number of training epochs.
        verbose_interval (int): Print loss every N epochs.

    Returns:
        LaurentModel: Trained model.
        list: Loss history during training.
    """
    Q_coef = torch.tensor(filter_laurent_coefficients(Q.coef, deg))
    P_coef = torch.tensor(P.coef)
    # print(f"Q: {Q_coef.size(0)}")
    # print(f"P: {P_coef.size(0)}")
    if Q_coef.size(0) < deg * 2 + 1:
        Q_coef = pad_tensor(Q_coef, deg=deg * 2 + 1)
    if P_coef.size(0) < deg * 2 + 1:
        P_coef = pad_tensor(P_coef, deg=deg * 2 + 1)
    # print(f"Q: {Q_coef.size(0)}")
    # print(f"P: {P_coef.size(0)}")
  
    # Step 1: Get indices to optimize based on parity
    indices_to_optimize = filter_indices(deg)

    # Step 2: Build model
    model = LaurentModel(Q_coef, indices_to_optimize)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )

    # Step 3: Training loop
    loss_history = []

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        Q_current = model.forward()
        Q_x = construct_function_values(Q_current)
        P_x = construct_function_values(P_coef)

        loss = torch.sum((torch.abs(Q_x)**2 + torch.abs(P_x)**2 - 1) ** 2)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step(loss.item())

        loss_history.append(loss.item())

        # if epoch % 50 == 0:
        #     print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    return Laurent(model.forward().detach())


def filter_indices(deg):
    """
    Return indices corresponding to frequency components that match the parity of deg.

    Args:
        deg (int): Maximum degree/frequency considered.

    Returns:
        torch.Tensor: Indices of components with matching parity.
    """
    is_deg_even = (deg % 2 == 0)
    indices = []
    for k in range(-deg, deg + 1):
        if (k % 2 == 0) == is_deg_even:
            idx = k + deg  # map to array index
            indices.append(idx)
    return torch.tensor(indices)


def construct_function_values(coeffs, num_points=None):
    """
    Construct time-domain function values from Laurent coefficients using IFFT.

    Args:
        coeffs (torch.Tensor): Laurent coefficients of shape (2N+1,)
        num_points (int): Number of points in the output signal. If None, use 2N+1.

    Returns:
        torch.Tensor: Complex-valued function values in time domain.
    """
    N = (coeffs.shape[0] - 1) // 2
    if num_points is None:
        num_points = coeffs.shape[0]

    # Extend to zero-padded spectrum to avoid aliasing
    full_spectrum = torch.zeros(num_points, dtype=torch.complex64)
    full_spectrum[:N+1] = coeffs[N:]     # corresponds to n >= 0
    full_spectrum[-N:] += coeffs[:N]     # corresponds to n < 0

    # Compute inverse Fourier transform
    x_values = torch.fft.ifft(full_spectrum, norm='ortho') * (num_points ** 0.5)
    return x_values


class LaurentModel(nn.Module):
    """
    A model representing a Laurent polynomial where only certain coefficients are learnable.
    The rest are fixed based on initial values.
    """
    def __init__(self, Q_coef, Q_mask):
        super().__init__()
        self.fixed_Q = Q_coef.detach().clone()
        self.Q_learnable = nn.Parameter(Q_coef[Q_mask].detach().clone())
        self.Q_mask = Q_mask

    def forward(self):
        """
        Reconstruct the full coefficient vector with learned and fixed parts.
        """
        Q = self.fixed_Q.clone()
        Q[self.Q_mask] = self.Q_learnable
        return Q
    
def pad_tensor(tensor, deg):
    """
    If the length of `tensor` is less than `deg`, pad zeros at the beginning and end
    to make its length equal to `deg`. If the length is already >= `deg`, return the original tensor.
    
    :param tensor: Input 1D torch.Tensor
    :param deg: Target minimum length
    :return: Padded 1D tensor
    """
    length = tensor.size(0)  # Get the current length of the tensor
    

    padding = deg - length
    
    # Distribute padding: more on front if odd number
    front_pad = (padding + 1) // 2
    back_pad = padding // 2
    
    # Concatenate zeros with the original tensor
    padded_tensor = torch.cat([
        torch.zeros(front_pad, dtype=tensor.dtype, device=tensor.device),  # Front padding
        tensor,
        torch.zeros(back_pad, dtype=tensor.dtype, device=tensor.device)   # Back padding
    ])
    
    return padded_tensor