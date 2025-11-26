import numpy as np
import pytest
from typing import Tuple


def FEM_2D_quad8_physical_gradient_CC0_H1_T3(
    node_coords: np.ndarray,
    node_values: np.ndarray,
    xi,
    eta
) -> np.ndarray:
    """
    Compute the physical (x, y) gradient of a scalar field for a quadratic
    8-node quadrilateral (Q8) element at one or more natural coordinates (ξ, η).

    This function evaluates shape function derivatives, forms the Jacobian
    from nodal coordinates, and maps natural derivatives to the physical domain.

    Parameters
    ----------
    node_coords : np.ndarray
        Nodal coordinates of the Q8 element.
        Shape: (8, 2). Each row corresponds to a node, with columns [x, y].
    node_values : np.ndarray
        Scalar nodal values associated with the element.
        Shape: (8,).
    xi : float or np.ndarray
        ξ-coordinate(s) of evaluation point(s) in the reference domain [-1, 1].
        Can be a scalar or array-like of shape (n_pts,).
    eta : float or np.ndarray
        η-coordinate(s) of evaluation point(s) in the reference domain [-1, 1].
        Can be a scalar or array-like of shape (n_pts,).

    Returns
    -------
    grad_phys : np.ndarray
        Physical gradient of the scalar field at each evaluation point.
        Shape: (2, n_pts), where rows correspond to [∂u/∂x, ∂u/∂y]
        and column j corresponds to point (xi[j], eta[j]).

    Notes
    -----
    This function uses standard 8-node quadrilateral (Q8) shape functions:

        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)

    Derivatives with respect to ξ and η are computed for each node in this order:
        [N1, N2, N3, N4, N5, N6, N7, N8]

    Node ordering (must match both `node_coords` and `node_values`):
        1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
        5: ( 0, -1),  6: ( 1,  0),  7: ( 0,  1),  8: (-1,  0)

    """

    # ---- Inner helper: shape functions + derivatives
    def _quad8_shape_functions_and_derivatives(xi_eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xi_eta = np.atleast_2d(xi_eta).astype(float)
        x = xi_eta[:, 0]
        y = xi_eta[:, 1]
        one = 1.0
        xm, xp = (one - x), (one + x)
        ym, yp = (one - y), (one + y)
        x2, y2 = x * x, y * y

        n = xi_eta.shape[0]
        N = np.empty((n, 8, 1), dtype=float)
        dN = np.empty((n, 8, 2), dtype=float)

        # Shape functions
        N[:, 0, 0] = -0.25 * xm * ym * (one + x + y)
        N[:, 1, 0] =  0.25 * xp * ym * (x - y - one)
        N[:, 2, 0] =  0.25 * xp * yp * (x + y - one)
        N[:, 3, 0] =  0.25 * xm * yp * (y - x - one)
        N[:, 4, 0] =  0.5  * (one - x2) * ym
        N[:, 5, 0] =  0.5  * xp * (one - y2)
        N[:, 6, 0] =  0.5  * (one - x2) * yp
        N[:, 7, 0] =  0.5  * xm * (one - y2)

        # Derivatives [∂/∂ξ, ∂/∂η]
        dN[:, 0, 0] =  0.25 * ym * (2.0 * x + y)
        dN[:, 0, 1] =  0.25 * xm * (x + 2.0 * y)
        dN[:, 1, 0] =  0.25 * ym * (2.0 * x - y)
        dN[:, 1, 1] =  0.25 * xp * (2.0 * y - x)
        dN[:, 2, 0] =  0.25 * yp * (2.0 * x + y)
        dN[:, 2, 1] =  0.25 * xp * (2.0 * y + x)
        dN[:, 3, 0] =  0.25 * yp * (2.0 * x - y)
        dN[:, 3, 1] =  0.25 * xm * (2.0 * y - x)
        dN[:, 4, 0] = -x * ym
        dN[:, 4, 1] = -0.5 * (one - x2)
        dN[:, 5, 0] =  0.5 * (one - y2)
        dN[:, 5, 1] = -(one + x) * y
        dN[:, 6, 0] = -x * yp
        dN[:, 6, 1] =  0.5 * (one - x2)
        dN[:, 7, 0] = -0.5 * (one - y2)
        dN[:, 7, 1] = -(one - x) * y

        return N, dN

    # ---- Main computation
    xi  = np.asarray(xi,  dtype=float).ravel()
    eta = np.asarray(eta, dtype=float).ravel()
    pts = np.column_stack([xi, eta])
    _, dN = _quad8_shape_functions_and_derivatives(pts)

    vals = np.asarray(node_values, dtype=float).reshape(8)
    n_pts = pts.shape[0]
    grad_phys = np.empty((2, n_pts), dtype=float)
    NC_T = node_coords.T  # (2,8)

    for p in range(n_pts):
        dN_p = dN[p]               # (8,2)
        grad_nat_p = dN_p.T @ vals # (2,)
        J_p = NC_T @ dN_p          # (2,2)
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
    task_id = "FEM_2D_quad8_physical_gradient_CC0_H1_T3"
    task_short_description = "returns the gradient of the shape function in physical coordinates for an 8 node quad"
    created_date = "2025-09-23"
    created_by = "elejeune11"
    main_fcn = FEM_2D_quad8_physical_gradient_CC0_H1_T3
    required_imports = ["import numpy as np", "import pytest", "from typing import Callable, Tuple"]
    fcn_dependencies = []
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
    return {
        "task_id": task_id,
        "task_short_description": task_short_description,
        "created_date": created_date,
        "created_by": created_by,
        "main_fcn": main_fcn,
        "required_imports": required_imports,
        "fcn_dependencies": fcn_dependencies,
        "reference_verification_inputs": reference_verification_inputs,
        "test_cases": test_cases,
        # "python_version": "version_number",
        # "package_versions": {"numpy": "version_number", },
    }