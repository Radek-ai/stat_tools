"""
Shared statistical helper functions.

These are used across pages (results analysis, balance plots) and CLI-style
diagnostics to reduce duplication and keep calculations consistent.
"""

from __future__ import annotations

from typing import Optional, Sequence, Callable, Any

import numpy as np
import pandas as pd


def smd(x1: pd.Series | np.ndarray, x2: pd.Series | np.ndarray) -> float:
    """
    Standardized Mean Difference (absolute).

    Returns NaN if either sample has <2 observations or pooled variance is zero.
    """
    if len(x1) < 2 or len(x2) < 2:
        return np.nan

    # Support both Series and ndarray inputs
    x1s = pd.Series(x1).astype(float)
    x2s = pd.Series(x2).astype(float)

    pooled = np.sqrt((x1s.var(ddof=1) + x2s.var(ddof=1)) / 2)
    return abs(x1s.mean() - x2s.mean()) / pooled if pooled else np.nan


def cuped_adjust(pre: pd.Series, post: pd.Series) -> Optional[pd.Series]:
    """
    CUPED adjustment.

    Returns adjusted post series, or None if var(pre)==0 (no adjustment possible).
    """
    X = pre.values
    Y = post.values

    var_x = np.var(X, ddof=1)
    if var_x == 0:
        return None

    theta = np.cov(Y, X, ddof=1)[0, 1] / var_x
    return post - theta * (pre - X.mean())


def pairwise_matrix(groups: Sequence[Any], fn: Callable[[Any, Any], float]) -> pd.DataFrame:
    """Create a full pairwise matrix (diagonal NaN) for a function over group pairs."""
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for g1 in groups:
        for g2 in groups:
            if g1 != g2:
                mat.loc[g1, g2] = fn(g1, g2)
    return mat

