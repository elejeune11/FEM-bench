import numpy as np
import pytest
from typing import Tuple
def quad8_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of quadratic (8-node) quadrilateral shape functions and derivatives.
    Parameters
    ----------
    xi : np.ndarray
        Natural coordinates (ξ, η) in the reference square.
    Returns
    -------
    N : np.ndarray
        Shape functions evaluated at the input points. Shape: (n, 8, 1).
        Node order: [N1, N2, N3, N4, N5, N6, N7, N8].
    dN_dxi : np.ndarray
        Partial derivatives w.r.t. (ξ, η). Shape: (n, 8, 2).
        Columns correspond to [∂()/∂ξ, ∂()/∂η] in the same node order.
    Raises
    ------
    ValueError
        If `xi` is not a NumPy array.
        If `xi` has shape other than (2,) or (n, 2).
        If `xi` contains non-finite values (NaN or Inf).
    Notes
    -----
    Shape functions:
        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)
    """
    if not isinstance(xi, np.ndarray):
        raise ValueError("xi must be a NumPy array.")
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError("1D xi must have shape (2,).")
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError("2D xi must have shape (n, 2).")
    else:
        raise ValueError("xi must have shape (2,) or (n, 2).")
    if not np.all(np.isfinite(xi)):
        raise ValueError("xi must contain finite values.")
    x = xi[:, 0].astype(float, copy=False)
    y = xi[:, 1].astype(float, copy=False)
    one = 1.0
    xm = (one - x)        # (1 - ξ)
    xp = (one + x)        # (1 + ξ)
    ym = (one - y)        # (1 - η)
    yp = (one + y)        # (1 + η)
    x2 = x * x
    y2 = y * y
    n = xi.shape[0]
    N = np.empty((n, 8, 1), dtype=float)
    dN = np.empty((n, 8, 2), dtype=float)
    N[:, 0, 0] = -0.25 * xm * ym * (one + x + y)              # N1
    N[:, 1, 0] =  0.25 * xp * ym * (x - y - one)              # N2
    N[:, 2, 0] =  0.25 * xp * yp * (x + y - one)              # N3
    N[:, 3, 0] =  0.25 * xm * yp * (y - x - one)              # N4
    N[:, 4, 0] =  0.5  * (one - x2) * ym                      # N5
    N[:, 5, 0] =  0.5  * xp * (one - y2)                      # N6
    N[:, 6, 0] =  0.5  * (one - x2) * yp                      # N7
    N[:, 7, 0] =  0.5  * xm * (one - y2)                      # N8
    dN[:, 0, 0] =  0.25 * ym * (2.0 * x + y)                  # dN1/dξ
    dN[:, 0, 1] =  0.25 * xm * (x + 2.0 * y)                  # dN1/dη
    dN[:, 1, 0] =  0.25 * ym * (2.0 * x - y)                  # dN2/dξ
    dN[:, 1, 1] =  0.25 * xp * (2.0 * y - x)                  # dN2/dη
    dN[:, 2, 0] =  0.25 * yp * (2.0 * x + y)                  # dN3/dξ
    dN[:, 2, 1] =  0.25 * xp * (2.0 * y + x)                  # dN3/dη
    dN[:, 3, 0] =  0.25 * yp * (2.0 * x - y)                  # dN4/dξ
    dN[:, 3, 1] =  0.25 * xm * (2.0 * y - x)                  # dN4/dη
    dN[:, 4, 0] = -x * ym                                     # dN5/dξ
    dN[:, 4, 1] = -0.5 * (one - x2)                           # dN5/dη
    dN[:, 5, 0] =  0.5 * (one - y2)                           # dN6/dξ
    dN[:, 5, 1] = -(one + x) * y                              # dN6/dη
    dN[:, 6, 0] = -x * yp                                     # dN7/dξ
    dN[:, 6, 1] =  0.5 * (one - x2)                           # dN7/dη
    dN[:, 7, 0] = -0.5 * (one - y2)                           # dN8/dξ
    dN[:, 7, 1] = -(one - x) * y                              # dN8/dη
    return N, dN
def compute_physical_gradient_quad8(node_coords: np.ndarray, node_values: np.ndarray, xi, eta) -> np.ndarray:
    """
    Compute the physical (x,y) gradient of a scalar field for a quadratic
    8-node quadrilateral (Q8) at one or more natural-coordinate points (xi, eta).
    Steps:
      1) evaluate Q8 shape-function derivatives at (xi, eta),
      2) form the Jacobian from nodal coordinates,
      3) build the natural gradient from nodal values,
      4) map to physical coordinates using the Jacobian.
    Parameters
    ----------
    node_coords : (8,2)
        Physical coordinates of the Q8 nodes.
    node_values : (8,)
        Scalar nodal values.
    xi, eta : scalar or array-like (n_pts,)
        Natural coordinates of evaluation points.
    Assumptions / Conventions
    -------------------------
    Uses the Q8 shape functions exactly as in
        `quad8_shape_functions_and_derivatives` with natural domain [-1, 1]^2.
    Expected node ordering (must match both `node_coords` and the shape functions):
        1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
        5: ( 0, -1),  6: ( 1,  0),  7: ( 0,  1),  8: (-1,  0)
    Passing nodes in a different order will produce incorrect results.
    Returns
    -------
    grad_phys : (