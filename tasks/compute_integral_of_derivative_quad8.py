import numpy as np
import pytest
from typing import Tuple


def quad_quadrature_2D(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return Gauss–Legendre quadrature points and weights for the
    reference square Q = { (xi, eta) : -1 <= xi <= 1, -1 <= eta <= 1 }.

    Supported rules (tensor products of 1D Gauss–Legendre):
      - 1 point  : 1×1   (exact for polynomials with degree ≤1 in each variable)
      - 4 points : 2×2   (exact for degree ≤3 in each variable)
      - 9 points : 3×3   (exact for degree ≤5 in each variable)

    Parameters
    ----------
    num_pts : int
        Total number of quadrature points (1, 4, or 9).

    Returns
    -------
    points : (num_pts, 2) float64 ndarray
        Quadrature points [xi, eta] on the reference square.
    weights : (num_pts,) float64 ndarray
        Quadrature weights corresponding to `points`. The sum of weights
        equals the area of Q, which is 4.0.

    Raises
    ------
    ValueError
        If `num_pts` is not one of {1, 4, 9}.
    """
    if num_pts not in (1, 4, 9):
        raise ValueError("num_pts must be one of {1, 4, 9}.")

    # Select 1D rule
    if num_pts == 1:
        nodes_1d = np.array([0.0], dtype=np.float64)
        w_1d = np.array([2.0], dtype=np.float64)
    elif num_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        nodes_1d = np.array([-a, +a], dtype=np.float64)
        w_1d = np.array([1.0, 1.0], dtype=np.float64)
    else:  # num_pts == 9
        b = np.sqrt(3.0 / 5.0)
        nodes_1d = np.array([-b, 0.0, +b], dtype=np.float64)
        w_1d = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0], dtype=np.float64)

    # Tensor product (xi varies fastest)
    XI, ETA = np.meshgrid(nodes_1d, nodes_1d, indexing="xy")
    points = np.column_stack([XI.ravel(order="C"), ETA.ravel(order="C")]).astype(np.float64, copy=False)

    WXI, WETA = np.meshgrid(w_1d, w_1d, indexing="xy")
    weights = (WXI * WETA).ravel(order="C").astype(np.float64, copy=False)

    return points, weights


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


def compute_integral_of_derivative_quad8(
    node_coords: np.ndarray,
    node_values: np.ndarray,
    num_gauss_pts: int
) -> np.ndarray:
    """
    ∫_Ω (∇u) dΩ for a single quadratic quadrilateral element (scalar field only).

    Summary
    -------
    Computes the area integral of the physical gradient of a scalar field u
    over a Q8 8-node quadrilateral using Gauss–Legendre quadrature
    on the reference square Q = [-1,1]×[-1,1], mapped isoparametrically to Ω.

    Dependencies (only)
    -------------------
    - quad8_shape_functions_and_derivatives(xi) -> (N: (n,8,1), dN: (n,8,2))
      used to assemble J(ξ,η) = node_coords^T @ dN and det(J).
    - compute_physical_gradient_quad8(node_coords, node_values, xi, eta) -> (2, n_pts)
      gives [∂u/∂x; ∂u/∂y] at quadrature points.
    - quad_quadrature_2D(num_pts) -> (points: (num_pts,2), weights: (num_pts,))
      Gauss-Legendre points/weights for num_pts ∈ {1,4,9}.

    Parameters
    ----------
    node_coords : (8,2) float
        Physical coordinates in Q8 node order:
          1: (-1,-1), 2: ( 1,-1), 3: ( 1, 1), 4: (-1, 1),
          5: ( 0,-1), 6: ( 1, 0), 7: ( 0, 1), 8: (-1, 0)
    node_values : (8,) or (8,1) float
        Scalar nodal values for u. (If (8,1), it is squeezed.)
    num_gauss_pts : {1,4,9}
        Total quadrature points (1×1, 2×2, or 3×3).

    Returns
    -------
    integral : (2,) float
        [∫_Ω ∂u/∂x dΩ, ∫_Ω ∂u/∂y dΩ].
    """
    NC = np.asarray(node_coords, dtype=float)        # (8,2)
    vals = np.asarray(node_values, dtype=float).reshape(8)  # (8,)

    # Quadrature on reference square
    pts, w = quad_quadrature_2D(num_gauss_pts)       # pts: (n_pts,2), w: (n_pts,)
    xi, eta = pts[:, 0], pts[:, 1]

    # det(J) at all points (vectorized)
    _, dN = quad8_shape_functions_and_derivatives(pts)   # (n_pts, 8, 2)
    J_all = NC.T[None, :, :] @ dN                        # (n_pts, 2, 2)
    detJ  = np.linalg.det(J_all)                         # (n_pts,)

    # Physical gradients at all points (2, n_pts)
    grad_phys = compute_physical_gradient_quad8(NC, vals, xi, eta)

    # Weighted sum over points
    scale = (w * detJ)[None, :]                          # (1, n_pts)
    return (grad_phys * scale).sum(axis=1)               # (2,)


def compute_integral_of_derivative_quad8_all_ones(node_coords, node_values, num_gauss_pts):
    """
    Expected-failure stub: ignores inputs and returns [1, 1].
    Useful as a trivial negative control.
    """
    return np.ones(2, dtype=float)


def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Analytic check with identity mapping (reference element = physical element).
    When x ≡ ξ and y ≡ η, the Jacobian is J = I with det(J) = 1, so the routine
    integrates the physical gradient directly over Q = [-1, 1] × [-1, 1].
    Using u(x, y) = x^3 + y^3 gives ∇u = [3x^2, 3y^2] and the exact integrals on Q are [4, 4].
    A 2×2 Gauss rule (num_gauss_pts=4) is exact for this case; the test checks we recover [4, 4].
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
    Analytic check with an affine geometric map and a linear field. We map the
    reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c. By placing
    the eight node quad mid-edge nodes at the arithmetic midpoints of the mapped corners, the
    isoparametric geometry is exactly affine, so the Jacobian is constant (J = A,
    det(J) = det(A)). For the linear scalar field u(x, y) = α + βx + γy, the physical
    gradient is constant ∇u = [β, γ], hence ∫_Ω ∇u dΩ = [β, γ] · Area(Ω). The area follows
    from the mapping: Area(Ω) = ∫_Q det(J) dQ = det(A) · Area(Q) = 4 · |det(A)|, so
    the exact result is [β, γ] · (4 · |det(A)|). Test to make sure the function matches
    this analytical solution.
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
    Test quadrature-order sensitivity on a deliberately curved, asymmetric mapping.
    One approach is to keep the four corners on the reference square but displace the mid-edge
    nodes asymmetrically, inducing a non-affine geometry (spatially varying J).
    With fixed, non-symmetric nodal values, the FE integrand becomes high-order in (ξ, η), 
    so a 3×3 rule should not coincide with 2×2 or 1×1.
    The test asserts that increasing the rule to 3×3 changes the result.
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
    task_id = "compute_integral_of_derivative_quad8"
    task_short_description = "returns the integral of the gradient over a single 8 node quad element"
    created_date = "2025-09-23"
    created_by = "elejeune11"
    main_fcn = compute_integral_of_derivative_quad8
    required_imports = ["import numpy as np", "import pytest", "from typing import Tuple"]
    fcn_dependencies = [quad_quadrature_2D, quad8_shape_functions_and_derivatives, compute_physical_gradient_quad8]
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
