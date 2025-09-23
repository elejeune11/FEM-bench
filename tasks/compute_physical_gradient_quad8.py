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
        - Shape (2,) for a single point, or (n, 2) for a batch of points.
        - Components must be finite (no NaN or Inf). Domain is typically [-1, 1]^2.

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
    # --- Validation & promotion to (n, 2)
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

    # --- Aliases
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

    # --- Shape functions
    N[:, 0, 0] = -0.25 * xm * ym * (one + x + y)              # N1
    N[:, 1, 0] =  0.25 * xp * ym * (x - y - one)              # N2
    N[:, 2, 0] =  0.25 * xp * yp * (x + y - one)              # N3
    N[:, 3, 0] =  0.25 * xm * yp * (y - x - one)              # N4
    N[:, 4, 0] =  0.5  * (one - x2) * ym                      # N5
    N[:, 5, 0] =  0.5  * xp * (one - y2)                      # N6
    N[:, 6, 0] =  0.5  * (one - x2) * yp                      # N7
    N[:, 7, 0] =  0.5  * xm * (one - y2)                      # N8

    # --- Derivatives: columns [∂/∂ξ, ∂/∂η]
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


def compute_physical_gradient_quad8(
    node_coords: np.ndarray,
    node_values: np.ndarray,
    xi,
    eta
) -> np.ndarray:
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
    grad_phys : (2, n_pts)
        Rows are [∂u/∂x, ∂u/∂y] at each point.
        Column j corresponds to the j-th input point (xi[j], eta[j]).
    """
    xi  = np.asarray(xi,  dtype=float).ravel()
    eta = np.asarray(eta, dtype=float).ravel()
    pts = np.column_stack([xi, eta])                # (n_pts, 2)

    _, dN = quad8_shape_functions_and_derivatives(pts)  # dN: (n_pts, 8, 2)

    vals = np.asarray(node_values, dtype=float).reshape(8)  # (8,)
    n_pts = pts.shape[0]
    grad_phys = np.empty((2, n_pts), dtype=float)

    NC_T = node_coords.T  # (2, 8)
    for p in range(n_pts):
        dN_p = dN[p]                     # (8, 2)
        grad_nat_p = dN_p.T @ vals       # (2,)
        J_p = NC_T @ dN_p                # (2, 2)
        grad_phys[:, p] = np.linalg.inv(J_p).T @ grad_nat_p

    return grad_phys


def compute_physical_gradient_quad8_all_ones(node_coords, node_values, xi, eta):
    """
    Always-one gradient.
    """
    xi = np.asarray(xi, float).ravel()
    return np.ones((2, xi.size), dtype=float)


def test_q8_gradient_identity_mapping_matches_quadratic_analytic(fcn):
    """
    Test the identity isoparametric mapping (x=ξ, y=η).
    For any example quadratic u(ξ,η)=a+bξ+cη+dξ²+eξη+fη²,
    the physical gradient equals the analytic gradient.
    """
    # Q8 node order: 4 corners then 4 mids
    nodes_nat = np.array([
        [-1.0, -1.0],  # 1
        [ 1.0, -1.0],  # 2
        [ 1.0,  1.0],  # 3
        [-1.0,  1.0],  # 4
        [ 0.0, -1.0],  # 5
        [ 1.0,  0.0],  # 6
        [ 0.0,  1.0],  # 7
        [-1.0,  0.0],  # 8
    ], dtype=float)

    # Identity mapping: physical coords == natural coords
    node_coords = nodes_nat.copy()

    # Quadratic polynomial in (ξ,η)
    a, b, c, d, e, f = 0.1, -0.7, 1.2, 0.4, -0.3, 0.6
    def u(xi, eta):  # value at a point
        return a + b*xi + c*eta + d*xi*xi + e*xi*eta + f*eta*eta
    def grad_u(xi, eta):  # analytic gradient [du/dx, du/dy] = [du/dξ, du/dη]
        return np.array([b + 2*d*xi + e*eta, c + e*xi + 2*f*eta], dtype=float)

    # Nodal values (evaluate u at the 8 nodes)
    node_values = np.array([u(xi, eta) for (xi, eta) in nodes_nat], dtype=float)

    # Deterministic interior sample points
    pts = np.array([
        [ 0.0,  0.0],
        [ 0.5,  0.0],
        [ 0.0, -0.4],
        [-0.3,  0.6],
        [ 0.2,  0.2],
    ], dtype=float)
    xi_eval = pts[:, 0]
    eta_eval = pts[:, 1]

    # Compute physical gradient via the implementation
    grad_phys = fcn(node_coords, node_values, xi_eval, eta_eval)  # (2, n_pts)

    # Analytic gradient (since x=ξ,y=η)
    gx = b + 2*d*xi_eval + e*eta_eval
    gy = c + e*xi_eval + 2*f*eta_eval
    grad_exact = np.vstack([gx, gy])

    diff = np.abs(grad_phys - grad_exact)
    max_err = np.max(diff)
    assert np.allclose(grad_phys, grad_exact, atol=1e-12), f"Max grad error = {max_err:.3e}\nDiff:\n{diff}"


def test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant(fcn):
    """
    Test that under any isoparametric mapping, a linear physical field
    u(x,y)=α+βx+γy is reproduced exactly by quadratic quadrilateral elements.
    The physical gradient should be [β, γ]^T at all points.
    """
    # Curved (non-affine) mapping: keep corners, bow mids slightly
    node_coords = np.array([
        [-1.0, -1.0],  # 1
        [ 1.0, -1.0],  # 2
        [ 1.0,  1.0],  # 3
        [-1.0,  1.0],  # 4
        [ 0.0, -1.05], # 5 (bottom mid bowed)
        [ 1.08, 0.0],  # 6 (right mid bowed)
        [ 0.0,  1.10], # 7 (top mid bowed)
        [-1.02, 0.05], # 8 (left mid bowed)
    ], dtype=float)

    # Linear field in physical coords
    alpha, beta, gamma = 0.2, -0.6, 1.3
    def u_phys(x, y): return alpha + beta*x + gamma*y

    # Nodal values from physical coords
    node_values = np.array([u_phys(x, y) for (x, y) in node_coords], dtype=float)

    # Interior evaluation points in natural space
    pts = np.array([
        [ 0.0,  0.0],   # centroid
        [ 0.3, -0.4],
        [-0.6,  0.2],
        [ 0.5,  0.5],
        [-0.2,  0.1],
    ], dtype=float)
    xi_eval  = pts[:, 0]
    eta_eval = pts[:, 1]

    grad_phys = fcn(node_coords, node_values, xi_eval, eta_eval)  # (2, n_pts)

    # Expected constant gradient [β, γ] for all points
    grad_exact = np.repeat(np.array([[beta], [gamma]], dtype=float), repeats=pts.shape[0], axis=1)  # (2, n_pts)

    diff = np.abs(grad_phys - grad_exact)
    max_err = np.max(diff)
    assert np.allclose(grad_phys, grad_exact, atol=1e-11), f"Max grad error = {max_err:.3e}\nDiff:\n{diff}"


def task_info():
    task_id = "compute_physical_gradient_quad8"
    task_short_description = "returns the gradient of the shape function in physical coordinates for an 8 node quad"
    created_date = "2025-09-23"
    created_by = "elejeune11"
    main_fcn = compute_physical_gradient_quad8
    required_imports = ["import numpy as np", "import pytest", "from typing import Tuple"]
    fcn_dependencies = [quad8_shape_functions_and_derivatives]
    reference_verification_inputs = [
        # 1) Identity mapping (natural == physical), single interior point
        [
            np.array([  # node_coords_1
                [-1.0, -1.0],  # 1
                [ 1.0, -1.0],  # 2
                [ 1.0,  1.0],  # 3
                [-1.0,  1.0],  # 4
                [ 0.0, -1.0],  # 5
                [ 1.0,  0.0],  # 6
                [ 0.0,  1.0],  # 7
                [-1.0,  0.0],  # 8
            ], dtype=float),
            np.array([  # node_values_1 (arbitrary scalar values)
                0.9, -0.3, 1.1, -0.2, 0.4, -0.5, 0.7, -0.1
            ], dtype=float),
            np.array([0.2], dtype=float),    # xi_1
            np.array([-0.3], dtype=float),   # eta_1
        ],

        # 2) Stretched rectangle (affine), 3 interior points
        [
            np.array([  # node_coords_2 (corners stretched; mids are edge midpoints)
                [-2.0, -1.0],   # 1
                [ 2.0, -1.0],   # 2
                [ 2.0,  1.5],   # 3
                [-2.0,  1.5],   # 4
                [ 0.0, -1.0],   # 5
                [ 2.0,  0.25],  # 6
                [ 0.0,  1.5],   # 7
                [-2.0,  0.25],  # 8
            ], dtype=float),
            np.array([  # node_values_2
                2.0, 1.2, 0.3, -0.7, 1.5, 0.8, -0.2, 1.1
            ], dtype=float),
            np.array([-0.7, 0.0, 0.8], dtype=float),  # xi_2
            np.array([ 0.0, 0.2, 0.3], dtype=float),  # eta_2
        ],

        # 3) Mildly curved mapping (bowed mids), 2 points
        [
            np.array([  # node_coords_3
                [-1.0, -1.0],
                [ 1.0, -1.0],
                [ 1.0,  1.0],
                [-1.0,  1.0],
                [ 0.0, -1.08],   # bottom mid bowed
                [ 1.07,  0.0],   # right mid bowed
                [ 0.0,  1.10],   # top mid bowed
                [-1.05,  0.05],  # left mid bowed
            ], dtype=float),
            np.array([  # node_values_3
                0.2, -1.0, 1.8, 0.9, -0.4, 1.3, -0.6, 0.5
            ], dtype=float),
            np.array([ 0.3, -0.2], dtype=float),  # xi_3
            np.array([ 0.1,  0.4], dtype=float),  # eta_3
        ],

        # 4) Skewed/rotated parallelogram (affine-ish), 4 points
        [
            np.array([  # node_coords_4 (corners + edge midpoints)
                [0.0, 0.0],    # 1
                [2.1, 0.6],    # 2
                [2.7, 2.1],    # 3
                [0.5, 1.7],    # 4
                [1.05, 0.3],   # 5 (mid 1-2)
                [2.4, 1.35],   # 6 (mid 2-3)
                [1.6, 1.9],    # 7 (mid 3-4)
                [0.25, 0.85],  # 8 (mid 4-1)
            ], dtype=float),
            np.array([  # node_values_4
                0.0, 0.6, 1.1, -0.3, 0.2, 0.9, 0.1, -0.1
            ], dtype=float),
            np.array([-0.2, 0.6, 0.0, 0.9], dtype=float),  # xi_4
            np.array([ 0.5, -0.1, -0.3, 0.0], dtype=float),# eta_4
        ],

        # 5) Asymmetric curved mapping (distinct mid offsets), 5 points
        [
            np.array([  # node_coords_5
                [-1.0, -1.0],
                [ 1.0, -1.0],
                [ 1.0,  1.0],
                [-1.0,  1.0],
                [ 0.0, -1.12],  # bottom mid offset
                [ 1.10,  0.08], # right mid offset
                [ 0.02,  1.15], # top mid offset
                [-1.06, -0.03], # left mid offset
            ], dtype=float),
            np.array([  # node_values_5
                -0.8, 0.5, 0.7, -0.4, 1.0, -0.2, 0.3, -0.6
            ], dtype=float),
            np.array([-0.8, -0.2, 0.2, 0.6, 0.9], dtype=float),  # xi_5
            np.array([ 0.6, -0.4, 0.5, 0.1,-0.2], dtype=float),  # eta_5
        ],
    ]
    test_cases = [{"test_code": test_q8_gradient_identity_mapping_matches_quadratic_analytic, "expected_failures": [compute_physical_gradient_quad8_all_ones]},
                  {"test_code": test_q8_gradient_linear_physical_field_under_curved_mapping_is_constant, "expected_failures": [compute_physical_gradient_quad8_all_ones]},
                  ]
    return task_id, task_short_description, created_date, created_by, main_fcn, required_imports, fcn_dependencies, reference_verification_inputs, test_cases
