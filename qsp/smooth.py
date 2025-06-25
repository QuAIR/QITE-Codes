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

from typing import Callable, List, Tuple

import numpy as np


def expand_interval(a: float, b: float, width: float) -> Tuple[float, float]:
    """
    Expand the interval [a, b] by 'width' on both sides, clamped to [-pi, pi].
    """
    return (max(-np.pi, a - width), min(np.pi, b + width))

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Merge overlapping or touching intervals.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            # Overlaps or touches => merge
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

def smooth_step(x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """
    Vectorized smooth step polynomial s(t) = 3t^2 - 2t^3 over [x0, x1].
    Returns 0.0 if x <= x0, 1.0 if x >= x1, and a smooth polynomial in-between.
    """
    # For a range of x values, we compute t in [0,1]
    # then apply 3t^2 - 2t^3
    out = np.zeros_like(x, dtype=float)
    left_mask = x <= x0
    right_mask = x >= x1
    mid_mask = ~(left_mask | right_mask)  # in-between

    # mid_mask region => polynomial
    t = (x[mid_mask] - x0) / (x1 - x0)
    out[mid_mask] = 3.0 * t**2 - 2.0 * t**3

    # left_mask region => 0
    # right_mask region => 1
    out[right_mask] = 1.0
    return out

def build_exclusion_zones(
    f: Callable[[np.ndarray], np.ndarray],
    discontinuities: List[float],
    nonsmooth_intervals: List[Tuple[float, float]],
    transition_width: float
) -> List[Tuple[float, float]]:
    """
    Build and merge all intervals (exclusion zones) around:
    - Discontinuities (discrete points)
    - Nonsmooth intervals
    - Possibly around -pi and pi if boundary values differ.
    """
    # 1) Turn discontinuities into intervals of width = transition_width
    half = transition_width / 2.0
    zones = [(d - half, d + half) for d in discontinuities]

    # 2) Add expanded nonsmooth intervals
    for (start_ns, end_ns) in nonsmooth_intervals:
        zones.append(expand_interval(start_ns, end_ns, transition_width))

    # 3) Merge so we have a minimal set of disjoint exclusion zones
    zones = merge_intervals(zones)

    # 4) Handle periodic boundary if f(-pi) != f(pi)
    val_minus = f(np.array([-np.pi]))[0]
    val_plus = f(np.array([np.pi]))[0]
    if not np.isclose(val_minus, val_plus, atol=1e-7):
        # Force small intervals near -pi and pi
        zones.append(expand_interval(-np.pi, -np.pi, transition_width))
        zones.append(expand_interval(np.pi, np.pi, transition_width))
        zones = merge_intervals(zones)

    return zones

def build_inclusion_zones(exclusion_zones: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Given exclusion zones, build the complementary (inclusion) zones
    in [-pi, pi] where f is used as-is.
    """
    inclusion = []
    cursor = -np.pi
    for (start_excl, end_excl) in exclusion_zones:
        if cursor < start_excl:
            inclusion.append((cursor, start_excl))
        cursor = max(cursor, end_excl)
    if cursor < np.pi:
        inclusion.append((cursor, np.pi))
    return inclusion

def apply_boundary_condition(
    x_array: np.ndarray,
    out: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray]
) -> None:
    """
    Enforce F(-pi) = F(pi) exactly, by setting the values at -pi and +pi
    to a consistent boundary value if they differ.
    """
    out_minus = f(np.array([-np.pi]))[0]
    out_plus = f(np.array([np.pi]))[0]
    if not np.isclose(out_minus, out_plus, atol=1e-7):
        mask_left = np.isclose(x_array, -np.pi, atol=1e-12)
        mask_right = np.isclose(x_array, np.pi, atol=1e-12)
        boundary_val = 0.5 * (out_minus + out_plus)
        out[mask_left] = boundary_val
        out[mask_right] = boundary_val

def to_smooth(
    f: Callable[[np.ndarray], np.ndarray],
    discontinuities: List[float] = [],
    nonsmooth_intervals: List[Tuple[float, float]] = [],
    transition_width: float = 0.1
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Construct a smooth function F on [-pi, pi] matching f on the largest subset possible,
    ensuring F(-pi) = F(pi).

    Args:
        f:
            A function f: [-pi, pi] -> { z in C : |z| <= 1 }.
            Implemented as a callable that accepts a numpy array x and returns
            a numpy array of the same shape.
        discontinuities:
            A list of floats indicating points of discontinuity/singularity
            in [-pi, pi].
        nonsmooth_intervals:
            A list of (start, end) floats indicating intervals on which f
            is not smooth.
        transition_width:
            A float indicating how wide a transition region to use around
            each discontinuity or nonsmooth boundary to ensure smoothness.

    Returns:
        A callable F: [-pi, pi] -> { z in C : |z| <= 1 }, with F(-pi) = F(pi).
        F(x) = f(x) on most of [-pi, pi], except for small neighborhoods
        around discontinuities/nonsmooth intervals, where it transitions smoothly.
    """

    # Build exclusion and inclusion zones
    exclusion_zones = build_exclusion_zones(f, discontinuities, nonsmooth_intervals, transition_width)
    inclusion_zones = build_inclusion_zones(exclusion_zones)

    def F(x_array: np.ndarray) -> np.ndarray:
        # Handle scalar inputs uniformly
        scalar_input = False
        if isinstance(x_array, (int, float)):
            x_array = np.array([x_array], dtype=float)
            scalar_input = True
        elif isinstance(x_array, list):
            x_array = np.array(x_array, dtype=float)

        # Clamp all x_array to [-pi, pi] at once
        x_array_clamped = np.clip(x_array, -np.pi, np.pi)

        # Prepare output and 'filled' mask to mark exclusion zones
        out = np.zeros_like(x_array_clamped, dtype=np.complex128)
        filled = np.zeros_like(x_array_clamped, dtype=bool)

        # Process each exclusion zone in order
        for (start_excl, end_excl) in exclusion_zones:
            # Values of f at the boundaries
            val_left = f(np.array([start_excl]))[0]
            val_right = f(np.array([end_excl]))[0]

            # Identify points in this interval that have not been filled
            mask_excl = (
                (x_array_clamped >= start_excl) &
                (x_array_clamped <= end_excl) &
                (~filled)
            )

            # Calculate smooth step between boundaries in a vectorized way
            alpha = smooth_step(x_array_clamped[mask_excl], start_excl, end_excl)

            # Blend from val_left to val_right
            out[mask_excl] = (1 - alpha) * val_left + alpha * val_right

            # Mark these points as filled
            filled[mask_excl] = True

        # Fill the remaining (inclusion) points with f(x) directly (one vectorized call to f)
        mask_inclusion = (~filled)
        if np.any(mask_inclusion):
            out[mask_inclusion] = f(x_array_clamped[mask_inclusion])

        # Enforce the periodic boundary condition at -pi and pi
        apply_boundary_condition(x_array_clamped, out, f)

        return out[0] if scalar_input else out

    return F