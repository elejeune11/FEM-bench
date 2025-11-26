import numpy as np
import pytest
from typing import Tuple


def FEM_2D_quad8_integral_of_derivative_CC0_H3_T3(
    node_coords: np.ndarray,
    node_values: np.ndarray,
    num_gauss_pts: int
) -> np.ndarray:
    """
    Compute ∫_Ω (∇u) dΩ for a scalar field u defined over a quadratic
    8-node quadrilateral (Q8) finite element.

    The computation uses isoparametric mapping and Gauss–Legendre quadrature
    on the reference domain Q = [-1, 1] × [-1, 1].

    Parameters
    ----------
    node_coords : np.ndarray
        Physical coordinates of the Q8 element nodes.
        Shape: (8, 2). Each row is [x, y].
        Node ordering (must match both geometry and values):
            1: (-1, -1),  2: ( 1, -1),  3: ( 1,  1),  4: (-1,  1),
            5: ( 0, -1),  6: ( 1,  0),  7: ( 0,  1),  8: (-1,  0)
    node_values : np.ndarray
        Scalar nodal values of u. Shape: (8,) or (8, 1).
    num_gauss_pts : int
        Number of quadrature points to use: one of {1, 4, 9}.
        - 1 → 1×1 rule, exact for ≤1st-degree polynomials
        - 4 → 2×2 rule, exact for ≤3rd-degree
        - 9 → 3×3 rule, exact for ≤5th-degree

    Returns
    -------
    integral : np.ndarray
        The vector [∫_Ω ∂u/∂x dΩ, ∫_Ω ∂u/∂y dΩ].
        Shape: (2,).

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

    # --- Inner helper: Gauss–Legendre quadrature
    def _quad_quadrature_2D(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
        if num_pts not in (1, 4, 9):
            raise ValueError("num_gauss_pts must be one of {1, 4, 9}.")

        if num_pts == 1:
            nodes_1d = np.array([0.0])
            w_1d = np.array([2.0])
        elif num_pts == 4:
            a = 1.0 / np.sqrt(3.0)
            nodes_1d = np.array([-a, +a])
            w_1d = np.array([1.0, 1.0])
        else:  # 9 points
            b = np.sqrt(3.0 / 5.0)
            nodes_1d = np.array([-b, 0.0, +b])
            w_1d = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

        XI, ETA = np.meshgrid(nodes_1d, nodes_1d, indexing="xy")
        pts = np.column_stack([XI.ravel(), ETA.ravel()])
        WXI, WETA = np.meshgrid(w_1d, w_1d, indexing="xy")
        weights = (WXI * WETA).ravel()
        return pts, weights

    # --- Inner helper: Q8 shape functions and derivatives
    def _quad8_shape_functions_and_derivatives(xi_eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xi_eta = np.atleast_2d(xi_eta).astype(float)
        x, y = xi_eta[:, 0], xi_eta[:, 1]
        one = 1.0
        xm, xp = (one - x), (one + x)
        ym, yp = (one - y), (one + y)
        x2, y2 = x*x, y*y

        n = xi_eta.shape[0]
        N = np.empty((n, 8, 1))
        dN = np.empty((n, 8, 2))

        # Shape functions
        N[:, 0, 0] = -0.25 * xm * ym * (one + x + y)
        N[:, 1, 0] =  0.25 * xp * ym * (x - y - one)
        N[:, 2, 0] =  0.25 * xp * yp * (x + y - one)
        N[:, 3, 0] =  0.25 * xm * yp * (y - x - one)
        N[:, 4, 0] =  0.5  * (one - x2) * ym
        N[:, 5, 0] =  0.5  * xp * (one - y2)
        N[:, 6, 0] =  0.5  * (one - x2) * yp
        N[:, 7, 0] =  0.5  * xm * (one - y2)

        # Derivatives
        dN[:, 0, 0] =  0.25 * ym * (2*x + y)
        dN[:, 0, 1] =  0.25 * xm * (x + 2*y)
        dN[:, 1, 0] =  0.25 * ym * (2*x - y)
        dN[:, 1, 1] =  0.25 * xp * (2*y - x)
        dN[:, 2, 0] =  0.25 * yp * (2*x + y)
        dN[:, 2, 1] =  0.25 * xp * (2*y + x)
        dN[:, 3, 0] =  0.25 * yp * (2*x - y)
        dN[:, 3, 1] =  0.25 * xm * (2*y - x)
        dN[:, 4, 0] = -x * ym
        dN[:, 4, 1] = -0.5 * (one - x2)
        dN[:, 5, 0] =  0.5 * (one - y2)
        dN[:, 5, 1] = -(one + x) * y
        dN[:, 6, 0] = -x * yp
        dN[:, 6, 1] =  0.5 * (one - x2)
        dN[:, 7, 0] = -0.5 * (one - y2)
        dN[:, 7, 1] = -(one - x) * y

        return N, dN

    # --- Main computation
    NC = np.asarray(node_coords, dtype=float)
    vals = np.asarray(node_values, dtype=float).reshape(8)
    pts, weights = _quad_quadrature_2D(num_gauss_pts)

    _, dN = _quad8_shape_functions_and_derivatives(pts)  # (n_pts, 8, 2)
    n_pts = pts.shape[0]
    J_all = NC.T[None, :, :] @ dN  # (n_pts, 2, 2)
    detJ = np.linalg.det(J_all)    # (n_pts,)

    grad_phys = np.empty((2, n_pts))
    for p in range(n_pts):
        J_p = J_all[p]
        dN_p = dN[p]
        grad_nat_p = dN_p.T @ vals
        grad_phys[:, p] = np.linalg.inv(J_p).T @ grad_nat_p

    scale = (weights * detJ)[None, :]
    integral = (grad_phys * scale).sum(axis=1)
    return integral


def compute_integral_of_derivative_quad8_all_ones(node_coords, node_values, num_gauss_pts):
    """
    Expected-failure stub: ignores inputs and returns [1, 1].
    Useful as a trivial negative control.
    """
    return np.ones(2, dtype=float)


def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case,
    """
    NC = np.array([
        [-1.0, -1.0],  # N1
        [ 1.0, -1.0],  # N2
        [ 1.0,  1.0],  # N3
        [-1.0,  1.0],  # N4
        [ 0.0, -1.0],  # N5
        [ 1.0,  0.0],  # N6
        [ 0.0,  1.0],  # N7
        [-1.0,  0.0],  # N8
    ], dtype=float)

    x, y = NC[:, 0], NC[:, 1]
    node_vals = x**3 + y**3  # (8,)

    integral = fcn(NC, node_vals, num_gauss_pts=4)
    expected = np.array([4.0, 4.0])
    assert integral.shape == (2,)
    assert np.allclose(integral, expected, rtol=1e-13, atol=1e-13)


def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution.
    """
    A = np.array([[ 1.2,  0.4],
                  [-0.3,  1.5]], dtype=float)
    c = np.array([0.7, -1.1], dtype=float)

    # Map corners affinely
    corners_nat = np.array([[-1, -1],
                            [ 1, -1],
                            [ 1,  1],
                            [-1,  1]], dtype=float)
    corners_phys = (corners_nat @ A.T) + c

    # Mid-edge nodes as arithmetic midpoints (keeps mapping affine)
    mids_phys = np.vstack([
        0.5*(corners_phys[0] + corners_phys[1]),  # N5
        0.5*(corners_phys[1] + corners_phys[2]),  # N6
        0.5*(corners_phys[2] + corners_phys[3]),  # N7
        0.5*(corners_phys[3] + corners_phys[0]),  # N8
    ])
    NC = np.vstack([corners_phys, mids_phys])

    # Linear scalar field
    alpha, beta, gamma = 0.3, -1.7, 2.4
    x, y = NC[:, 0], NC[:, 1]
    node_vals = alpha + beta*x + gamma*y

    area = 4.0 * abs(np.linalg.det(A))
    expected = np.array([beta, gamma]) * area

    integral = fcn(NC, node_vals, num_gauss_pts=4)
    assert integral.shape == (2,)
    assert np.allclose(integral, expected, rtol=1e-13, atol=1e-13)


def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings.
    """
    # --- Arrange: curved, asymmetric geometry + fixed nodal values
    corners = np.array([
        [-1.0, -1.0],  # N1
        [ 1.0, -1.0],  # N2
        [ 1.0,  1.0],  # N3
        [-1.0,  1.0],  # N4
    ], dtype=float)
    mids = np.array([
        [ 0.10, -0.55],  # N5
        [ 0.85,  0.10],  # N6
        [ 0.00,  0.80],  # N7
        [-0.95, -0.05],  # N8
    ], dtype=float)
    NC = np.vstack([corners, mids])  # (8,2)

    node_vals = np.array(
        [0.23, -0.91, 1.42, -0.37, 0.71, -1.28, 0.63, 0.05],
        dtype=float
    )

    # --- Act: integrate with 1×1, 2×2, and 3×3 rules
    I1 = fcn(NC, node_vals, num_gauss_pts=1)
    I4 = fcn(NC, node_vals, num_gauss_pts=4)
    I9 = fcn(NC, node_vals, num_gauss_pts=9)

    # --- Assert: 3×3 should differ from at least one lower-order result
    assert I1.shape == (2,) and I4.shape == (2,) and I9.shape == (2,)

    diff_9_4 = np.max(np.abs(I9 - I4))
    diff_9_1 = np.max(np.abs(I9 - I1))
    tol = 1e-10

    assert (diff_9_4 > tol) or (diff_9_1 > tol)


def task_info():
    task_id = "FEM_2D_quad8_integral_of_derivative_CC0_H3_T3"
    task_short_description = "returns the integral of the gradient over a single 8 node quad element"
    created_date = "2025-09-23"
    created_by = "elejeune11"
    main_fcn = FEM_2D_quad8_integral_of_derivative_CC0_H3_T3
    required_imports = ["import numpy as np", "import pytest", "from typing import Callable, Tuple"]
    fcn_dependencies = []
    reference_verification_inputs = [
        # 1) Identity mapping (natural == physical)
        [
            np.array([
                [-1.0, -1.0],  # 1
                [ 1.0, -1.0],  # 2
                [ 1.0,  1.0],  # 3
                [-1.0,  1.0],  # 4
                [ 0.0, -1.0],  # 5
                [ 1.0,  0.0],  # 6
                [ 0.0,  1.0],  # 7
                [-1.0,  0.0],  # 8
            ], dtype=float),
            np.array([  # arbitrary scalar values
                0.9, -0.3, 1.1, -0.2, 0.4, -0.5, 0.7, -0.1
            ], dtype=float),
            1,  # num_gauss_pts
        ],

        # 2) Stretched rectangle (affine; mids are edge midpoints)
        [
            np.array([
                [-2.0, -1.0],   # 1
                [ 2.0, -1.0],   # 2
                [ 2.0,  1.5],   # 3
                [-2.0,  1.5],   # 4
                [ 0.0, -1.0],   # 5
                [ 2.0,  0.25],  # 6
                [ 0.0,  1.5],   # 7
                [-2.0,  0.25],  # 8
            ], dtype=float),
            np.array([
                2.0, 1.2, 0.3, -0.7, 1.5, 0.8, -0.2, 1.1
            ], dtype=float),
            4,
        ],

        # 3) Mildly curved mapping (bowed mids)
        [
            np.array([
                [-1.0, -1.0],
                [ 1.0, -1.0],
                [ 1.0,  1.0],
                [-1.0,  1.0],
                [ 0.0, -1.08],   # bottom mid bowed
                [ 1.07,  0.0],   # right mid bowed
                [ 0.0,  1.10],   # top mid bowed
                [-1.05,  0.05],  # left mid bowed
            ], dtype=float),
            np.array([
                0.2, -1.0, 1.8, 0.9, -0.4, 1.3, -0.6, 0.5
            ], dtype=float),
            9,
        ],

        # 4) Skewed/rotated parallelogram (affine-ish)
        [
            np.array([
                [0.0, 0.0],    # 1
                [2.1, 0.6],    # 2
                [2.7, 2.1],    # 3
                [0.5, 1.7],    # 4
                [1.05, 0.3],   # 5 (mid 1-2)
                [2.4, 1.35],   # 6 (mid 2-3)
                [1.6, 1.9],    # 7 (mid 3-4)
                [0.25, 0.85],  # 8 (mid 4-1)
            ], dtype=float),
            np.array([
                0.0, 0.6, 1.1, -0.3, 0.2, 0.9, 0.1, -0.1
            ], dtype=float),
            4,
        ],

        # 5) Asymmetric curved mapping (distinct mid offsets)
        [
            np.array([
                [-1.0, -1.0],
                [ 1.0, -1.0],
                [ 1.0,  1.0],
                [-1.0,  1.0],
                [ 0.0, -1.12],  # bottom mid offset
                [ 1.10,  0.08], # right mid offset
                [ 0.02,  1.15], # top mid offset
                [-1.06, -0.03], # left mid offset
            ], dtype=float),
            np.array([
                -0.8, 0.5, 0.7, -0.4, 1.0, -0.2, 0.3, -0.6
            ], dtype=float),
            9,
        ],
    ]
    test_cases = [{"test_code": test_integral_of_derivative_quad8_identity_cubic, "expected_failures": [compute_integral_of_derivative_quad8_all_ones]},
                  {"test_code": test_integral_of_derivative_quad8_affine_linear_field, "expected_failures": [compute_integral_of_derivative_quad8_all_ones]},
                  {"test_code": test_integral_of_derivative_quad8_order_check_asymmetric_curved, "expected_failures": [compute_integral_of_derivative_quad8_all_ones]},
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