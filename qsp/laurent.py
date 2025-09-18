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

import warnings
from collections import Counter
from copy import copy
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy
from numpy.polynomial.polynomial import Polynomial, polyfromroots

r"""
Definition of ``Laurent`` class and its functions
"""


__all__ = ['Laurent', 'ascending_coef', 'revise_tol', 'remove_abs_error', 
           'sqrt_generation', 'Q_generation', 'pair_generation', 'laurent_generator']


TOL = 1e-30 # the error tolerance for Laurent polynomial, default to be machinery


class Laurent(object):
    r"""Class for Laurent polynomial defined as 
    :math:`P:\mathbb{C}[X, X^{-1}] \to \mathbb{C} :x \mapsto \sum_{j = -L}^{L} p_j X^j`.
    
    Args:
        coef: list of coefficients of Laurent poly, arranged as :math:`\{p_{-L}, ..., p_{-1}, p_0, p_1, ..., p_L\}`.

    """
    def __init__(self, coef: np.ndarray) -> None:
        if not isinstance(coef, np.ndarray):
            coef = np.asarray(coef)
        coef = coef.astype('complex128')
        coef = remove_abs_error(np.squeeze(coef) if len(coef.shape) > 1 else coef)
        assert len(coef.shape) == 1 and coef.shape[0] % 2 == 1
        
        # if the first and last terms are both 0, remove them
        while len(coef) > 1 and coef[0] == coef[-1] == 0:
            coef = coef[1:-1]
        
        # decide degree of this poly
        L = (len(coef) - 1) // 2 if len(coef) > 1 else 0
        
        # rearrange the coef in order p_0, ..., p_L, p_{-L}, ..., p_{-1},
        #   then we can call ``poly_coef[i]`` to retrieve p_i, this order is for internal use only
        coef = coef.tolist()
        poly_coef = np.asarray(coef[L:] + coef[:L]).astype('complex128')
        
        self.deg = L
        self.__coef = poly_coef
        
    def __call__(self, X: Union[int, float, complex, np.ndarray]) -> complex:
        r"""Evaluate the value of P(X).
        """
        if (not isinstance(X, np.ndarray)) and X == 0:
            return self.__coef[0]

        deg = self.deg
        if isinstance(X, np.ndarray):
           coef = np.asarray(self.__coef)
           return np.sum([coef[i] * np.exp(1j * i * X / 2) for i in range(-deg, deg + 1)], axis=0)
        return sum(self.__coef[i] * np.exp(1j * i * X / 2) for i in range(-self.deg, self.deg + 1))
    
    @property
    def coef(self) -> np.ndarray:
        r"""The coefficients of this polynomial in ascending order (of indices).
        """
        return ascending_coef(self.__coef)
    
    @property
    def conj(self) -> 'Laurent':
        r"""The conjugate of this polynomial i.e. :math:`P(x) = \sum_{j = -L}^{L} p_{-j}^* X^j`.
        """
        coef = np.copy(self.__coef)
        for i in range(1, self.deg + 1):
            coef[i], coef[-i] = coef[-i], coef[i]
        coef = np.conj(coef)
        return Laurent(ascending_coef(coef))
    
    @property
    def roots(self) -> np.ndarray:
        r"""List of roots of this polynomial.
        """
        # create the corresponding (common) polynomial with degree 2L
        P = Polynomial(self.coef)
        roots = P.roots().tolist()
        return np.asarray(sorted(roots, key=lambda x: np.abs(x)))
    
    @property
    def norm(self) -> float:
        r"""The square sum of absolute value of coefficients of this polynomial.
        """
        return np.sum(np.square(np.abs(self.__coef)))
    
    @property
    def max_norm(self) -> float:
        r"""The maximum of absolute value of coefficients of this polynomial.
        """
        list_x = np.arange(-np.pi, np.pi + 0.005, 0.005)
        return max(np.abs(self(x)) for x in list_x)
    
    @property
    def parity(self) -> int:
        r""" Parity of this polynomial.
        """
        coef = np.copy(self.__coef)

        even = not any(i % 2 != 0 and coef[i] != 0 for i in range(-self.deg, self.deg + 1))
        odd = not any(i % 2 != 1 and coef[i] != 0 for i in range(-self.deg, self.deg + 1))

        if even:
            return 0
        return 1 if odd else None
    
    def __copy__(self) -> 'Laurent':
        r"""Copy of Laurent polynomial.
        """
        return Laurent(ascending_coef(self.__coef))
    
    def __add__(self, other: Any) -> 'Laurent':
        r"""Addition of Laurent polynomial.
        
        Args:
            other: scalar or a Laurent polynomial :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`.
        
        """
        coef = np.copy(self.__coef)

        if isinstance(other, (int, float, complex)):
            coef[0] += other

        elif isinstance(other, Laurent):
            if other.deg > self.deg:
                return other + self

            deg_diff = self.deg - other.deg

            # retrieve the coef of Q
            q_coef = other.coef
            q_coef = np.concatenate([q_coef[other.deg:], np.zeros(2 * deg_diff),
                                     q_coef[:other.deg]]).astype('complex128')
            coef += q_coef

        else:
            raise TypeError(
                f"does not support the addition between Laurent and {type(other)}.")

        return Laurent(ascending_coef(coef))
    
    def __mul__(self, other: Any) -> 'Laurent':
        r"""Multiplication of Laurent polynomial.
        
        Args:
            other: scalar or a Laurent polynomial :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`.
        
        """
        p_coef = np.copy(self.__coef)
        if isinstance(other, (int, float, complex, np.longdouble, np.clongdouble)):
            new_coef = p_coef * other

        elif isinstance(other, Laurent):
            # retrieve the coef of Q
            q_coef = other.coef.tolist()
            q_coef = np.asarray(q_coef[other.deg:] + q_coef[:other.deg]).astype('complex128')

            L = self.deg + other.deg # deg of new poly
            new_coef = np.zeros([2 * L + 1]).astype('complex128')

            # (P + Q)[X^n] = \sum_{j, k st. j + k = n} p_j q_k
            for j in range(-self.deg, self.deg + 1):
                for k in range(-other.deg, other.deg + 1):
                    new_coef[j + k] += p_coef[j] * q_coef[k]

        else:
            raise TypeError(
                f"does not support the multiplication between Laurent and {type(other)}.")

        return Laurent(ascending_coef(new_coef))
    
    def __sub__(self, other: Any) -> 'Laurent':
        r"""Subtraction of Laurent polynomial.
        
        Args:
            other: scalar or a Laurent polynomial :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`.
        
        """
        return self.__add__(other=other * -1)
    
    def __eq__(self, other: Any) -> bool:
        r"""Equality of Laurent polynomial.
        
        Args:
            other: a Laurent polynomial :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`.
        
        """
        if isinstance(other, (int, float, complex)):
            p_coef = self.__coef
            constant_term = p_coef[0]
            return self.deg == 0 and np.abs(constant_term - other) < max(1e-3, TOL)

        elif isinstance(other, Laurent):
            p_coef = self.coef
            q_coef = other.coef
            return self.deg == other.deg and np.max(np.abs(p_coef - q_coef)) < max(1e-3, TOL)

        else:
            raise TypeError(
                f"does not support the equality between Laurent and {type(other)}.")
    
    def __str__(self) -> str:
        r"""Print of Laurent polynomial.
        
        """
        coef = np.around(self.__coef, 3)
        L = self.deg
        
        print_str = "info of this Laurent poly is as follows\n"
        print_str += f"   - constant: {coef[0]}    - degree: {L}\n"
        if L > 0:
            print_str += f"   - coef of terms from pos 1 to pos {L}: {coef[1:L + 1]}\n"
            print_str += f"   - coef of terms from pos -1 to pos -{L}: {np.flip(coef[L + 1:])}\n"
        return print_str
    
    def is_parity(self, p: int) -> Tuple[bool, complex]:
        r"""Whether this Laurent polynomial has parity :math:`p % 2`.
        
        Args:
            p: parity.
        
        Returns:
            contains the following elements:
            * whether parity is p % 2;
            * if not, then return the the (maximum) absolute coef term breaking such parity;
            * if not, then return the the (minimum) absolute coef term obeying such parity.
        
        """
        p %= 2
        coef = np.copy(self.__coef)

        disagree_coef = []
        agree_coef = []
        for i in range(-self.deg, self.deg + 1):
            c = coef[i]
            if i % 2 != p and c != 0:
                disagree_coef.append(c)
            elif i % 2 == p:
                agree_coef.append(c)

        return (False, max(np.abs(disagree_coef)), min(np.abs(agree_coef))) if disagree_coef else (True, None, None)
    
    def reduced_poly(self, target_deg: int) -> 'Laurent':
        r"""Generate :math:`P'(x) = \sum_{j = -D}^{D} p_j X^j`, where :math:`D \leq L` is `target_deg`.
        
        Args:
            target_deg: the degree of returned polynomial
        
        """
        coef = self.coef
        L = self.deg
        return Laurent(coef[L - target_deg:L + 1 + target_deg]) if target_deg <= L else Laurent(coef)


def revise_tol(t: float) -> None:
    r"""Revise the value of error tolerance `TOL`.
    
    Value below `TOL` may be considered as error.
    """
    global TOL
    assert t > 0
    TOL = t


def ascending_coef(coef: np.ndarray) -> np.ndarray:
    r"""Transform the coefficients of a polynomial in ascending order (of indices).
    
    Args:
        coef: list of coefficients arranged as :math:`\{ p_0, ..., p_L, p_{-L}, ..., p_{-1} \}`.
        
    Returns:
        list of coefficients arranged as :math:`\{ p_{-L}, ..., p_{-1}, p_0, p_1, ..., p_L \}`.
    
    """
    L = int((len(coef) - 1) / 2)
    coef = coef.tolist()
    return np.asarray(coef[L + 1:] + coef[:L + 1])


def remove_abs_error(data: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    r"""Remove the error in data array.
    
    Args:
        data: data array.
        tol: error tolerance.
        
    Returns:
        sanitized data.
        
    """
    data_len = len(data)
    tol = TOL if tol is None else tol

    for i in range(data_len):
        if np.abs(np.real(data[i])) < tol:
            data[i] = 1j * np.imag(data[i])
        elif np.abs(np.imag(data[i])) < tol:
            data[i] = np.real(data[i])
        
        if np.abs(data[i]) < tol:
            data[i] = 0
    return data


def _generate_key(xi: np.ndarray, precision: float) -> Tuple[float, float]:
    float_tol = max(precision, np.sqrt(TOL))
    digits = int(np.log10(1 / float_tol))
    return (xi.real.round(digits), xi.imag.round(digits))

def _search_inv_conj_pair(pool_roots: np.ndarray, precision: float = 1e-4) -> Tuple[List[complex], List[complex]]:
    counts, dict_roots = Counter(), {}

    for r in pool_roots:
        key = _generate_key(r, precision)
        counts[key] += 1
        if key not in dict_roots:
            dict_roots[key] = r

    Q_roots, inv_conj_roots = [], []
    excess_roots = []
    processed_keys = []
    
    for key, occurrence in counts.items():
        r = dict_roots[key]
        
        if np.abs(r * r.conj() - 1) < 1e-5:
            if occurrence % 2 == 0:
                list_roots = [r for _ in range(occurrence // 2)]
                Q_roots.extend(list_roots)
                inv_conj_roots.extend(list_roots)
            else:
                excess_roots.append([r] * occurrence)
            continue
        
        list_roots = [r for _ in range(occurrence)]
        
        inv_conj_xi = 1 / r.conj()
        inv_conj_key = _generate_key(inv_conj_xi, precision)
        if inv_conj_key in processed_keys:
            inv_conj_roots.extend(list_roots)
            continue
        
        processed_keys.append(key)
        Q_roots.extend(list_roots)
        
    if excess_roots:
        warnings.warn(
            f"{len(excess_roots)} excess roots found with precision {precision}", UserWarning)
        excess_Q_roots, excess_inv_conj_roots = _search_inv_conj_pair(np.asarray(excess_roots).flatten(), precision * 5)
        Q_roots.extend(excess_Q_roots)
        inv_conj_roots.extend(excess_inv_conj_roots)
    return Q_roots, inv_conj_roots
    
def sqrt_generation(A: Laurent, disp: float = 0, precision: float = 1e-4) -> Laurent:
    
    r"""Generate the "square root" of a Laurent polynomial :math:`A`.
    
    Args:
        A: a Laurent polynomial.
        disp: displacement that A adds, defaults to be 0. Useful when A(x) = 0 for lots of x.
        
    Returns:
        a Laurent polynomial :math:`Q` such that :math:`QQ^* = A`.
        
    Note:
        More details are in Lemma S1 of the paper https://arxiv.org/abs/2209.14278.  
    
    """
    disp = disp if disp > 0 else np.sqrt(TOL)
    
    origA, A = copy(A), A + disp
    roots = A.roots
    Q_roots, inv_roots = _search_inv_conj_pair(roots, precision=precision)

    # be careful that the number of filtered roots should be identical with that of the saved roots
    if len(Q_roots) != len(inv_roots):
        warnings.warn(
            "\nError occurred in square root decomposition of polynomial: " +
            f"# of total, saved and filtered roots are {len(roots)}, {len(Q_roots)}, {len(inv_roots)}." +
            "\n     Will force equal size of saved and filtered root list to mitigate the error")
        excess_roots, Q_roots = Q_roots[len(inv_roots):], Q_roots[:len(inv_roots)]
        excess_roots.sort(key=lambda x: np.real(x)) # sort by real part
        
        for i in range(len(excess_roots) // 2):
            Q_roots.append(excess_roots[2 * i])
            inv_roots.append(excess_roots[2 * i + 1])            
    inv_roots = np.asarray(inv_roots)

    # construct Q
    leading_coef = A.coef[-1]
    Q_coef = polyfromroots(Q_roots) * np.sqrt(leading_coef * np.prod(inv_roots))
    Q = Laurent(Q_coef)

    # final output test
    if Q * Q.conj != origA:
        warnings.warn(
            f"\ncomputation error: QQ* != A, check your code \n degree of Q: {Q.deg}, degree of A: {A.deg}")

    return Q


def Q_generation(P: Laurent, disp: float = 0, precision: float = 1e-4) -> Laurent:
    r"""Generate a Laurent complement for Laurent polynomial :math:`P`.
    
    Args:
        P: a Laurent poly with parity :math:`L` and degree :math:`L`.
        disp: displacement that taking square roots needs, defaults to be 0.
        
    Returns:
        a Laurent poly :math:`Q` st. :math:`PP^* + QQ^* = 1`, with parity :math:`L` and degree :math:`L`.
    
    """
    assert P.parity is not None and P.parity == P.deg % 2, \
        "this Laurent poly does not satisfy the requirement for parity"
    assert P.max_norm < 1, \
        f"the max norm {P.max_norm} of this Laurent poly should be smaller than 1"
    
    Q2 = P * P.conj * -1 + 1
    Q = sqrt_generation(Q2, disp, precision=precision)
        
    is_parity, max_diff, min_val = Q.is_parity(P.parity)
    if not is_parity:
        warnings.warn(
            f"\nQ's parity {Q.parity} does not agree with P's parity {P.parity}, max err is {max_diff}, min val is {min_val}")

    return Q


def pair_generation(f: Laurent) -> Laurent:
    r""" Generate Laurent pairs for Laurent polynomial :math:`f`.
    
    Args:
        f: a real-valued and even Laurent polynomial, with max_norm smaller than 1.
    
    Returns:
        Laurent polys :math:`P, Q` st. :math:`P = \sqrt{1 + f / 2}, Q = \sqrt{1 - f / 2}`.
    
    """
    assert f.max_norm < 1, \
        f"the max norm {f.max_norm} of this Laurent poly should be smaller than 1"
    assert f.parity == 0, \
        "the parity of this Laurent poly should be 0"
    expect_parity = (f.deg // 2) % 2
    P, Q = sqrt_generation((f + 1) * 0.5), sqrt_generation((f * (-1) + 1) * 0.5)

    is_parity, max_diff, min_val = P.is_parity(expect_parity)
    if not is_parity:
        warnings.warn(
            f"\nP's parity {P.parity} does not agree with {expect_parity}, max err is {max_diff}, min val is {min_val}", UserWarning)

    is_parity, max_diff, min_val = Q.is_parity(expect_parity)
    if not is_parity:
        warnings.warn(
            f"\nQ's parity {Q.parity} does not agree with {expect_parity}, max err is {max_diff}, min val is {min_val}", UserWarning)
    return P, Q


def func_fft(f: Callable[[np.ndarray], np.ndarray], N: int) -> np.ndarray:
    """Approximate the complex Fourier coefficients c_j for j = -N..N.

    This function computes the Fourier coefficients in the trigonometric form:
    sum_{j=-N}^N c_j e^{i j x}, for x in the interval [-π, π]. It uses SciPy's
    FFT to approximate these coefficients based on sampled values of the
    function f.

    Args:
        f: A function f(x) implemented via NumPy, returning complex
            values. The domain of f is assumed to be x ∈ [-π, π].
        N: Degree of truncation; 2N+1 coefficients are returned.

    Returns:
        The Fourier coefficients [c_{-N}, ..., c_{N}]. The entry c[j+N] 
            corresponds to the coefficient c_j in the sum.
    """
    # Choose a sample size M ≥ 2N+1. A simple choice is 2*(2N+1) for oversampling.
    M = 2 * (2*N + 1)

    # Sample points x_m in [-π, π), equally spaced
    x_vals = np.linspace(-np.pi, np.pi, M, endpoint=False)
    
    # Evaluate f at these sample points
    f_samples = f(x_vals)

    # Compute the discrete Fourier transform of these samples
    F = scipy.fft.fft(f_samples)

    # Prepare storage for the coefficients c_{-N..N}
    coef = np.zeros(2*N + 1, dtype=complex)

    # Extract each coefficient using the relationship c_j = [(-1)^j / M] * F[j mod M]
    for j in range(-N, N + 1):
        index = j % M  # wraps negative j around to 0..M-1
        coef[j + N] = ((-1)**j / M) * F[index]
    return coef


def laurent_generator(f: Callable[[np.ndarray], np.ndarray], deg: int) -> Laurent:
    r"""Generate a Laurent polynomial (with :math:`X = e^{ix / 2}`) approximating `fn`.
    
    This function generates a Laurent polynomial from the Fourier coefficients
    of a function f(x) on the interval [-π, π]. The coefficients are computed
    using scipy fft.

    Args:
        fn: function to be approximated.
        deg: degree of Laurent poly.

    Returns:
        a Laurent polynomial approximating `fn` in interval [-π, π] with degree `deg`.
    
    """
    assert deg >= 0 and deg % 2 == 0, \
        f"Degree must be a non-negative even number, got {deg}."
    coef = func_fft(f, deg // 2)
    coef = np.asarray([coef[i // 2] if i % 2 == 0 else 0 for i in range(deg * 2 + 1)])
    return Laurent(coef)


def deg_finder(fn: Callable[[np.ndarray], np.ndarray], 
               delta: Optional[float] = 0.00001 * np.pi, l: Optional[float] = np.pi) -> int:
    r"""Find a degree such that the Laurent polynomial generated from `laurent_generator` has max_norm smaller than 1.
    
    Args:
        fn: function to be approximated.
        dx: sampling frequency of data points, defaults to be :math:`0.00001 \pi`.
        L: half of approximation width, defaults to be :math:`\pi`.
    
    Returns:
        the degree of approximation:
        
    Note:
        used to fix the problem of function `laurent_generator`.
    
    """
    deg = 50
    acc = 1
    P = laurent_generator(fn, delta, deg, l)
    while P.max_norm > 1:
        deg += acc * 50
        P = laurent_generator(fn, delta, deg, l)
        acc += 1
        assert deg <= 10000, "degree too large"
    return deg
