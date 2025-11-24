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


def quad8_shape_functions_and_derivatives_no_error(xi):
    """
    Buggy-style version for Q8 (no input validation).
    Accepts xi with shape (2,) or (n,2) and returns:
      N : (n, 8, 1)
      dN: (n, 8, 2)
    """
    # Force input to at least 2D without validation
    xi = np.atleast_2d(xi).astype(float)
    x = xi[:, 0]
    y = xi[:, 1]

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

    # Shape functions (serendipity Q8)
    N[:, 0, 0] = -0.25 * xm * ym * (one + x + y)      # N1
    N[:, 1, 0] =  0.25 * xp * ym * (x - y - one)      # N2
    N[:, 2, 0] =  0.25 * xp * yp * (x + y - one)      # N3
    N[:, 3, 0] =  0.25 * xm * yp * (y - x - one)      # N4
    N[:, 4, 0] =  0.5  * (one - x2) * ym              # N5 (mid-side)
    N[:, 5, 0] =  0.5  * xp * (one - y2)              # N6 (mid-side)
    N[:, 6, 0] =  0.5  * (one - x2) * yp              # N7 (mid-side)
    N[:, 7, 0] =  0.5  * xm * (one - y2)              # N8 (mid-side)

    # Derivatives: columns [∂/∂ξ, ∂/∂η]
    dN[:, 0, 0] =  0.25 * ym * (2.0 * x + y)          # dN1/dξ
    dN[:, 0, 1] =  0.25 * xm * (x + 2.0 * y)          # dN1/dη

    dN[:, 1, 0] =  0.25 * ym * (2.0 * x - y)          # dN2/dξ
    dN[:, 1, 1] =  0.25 * xp * (2.0 * y - x)          # dN2/dη

    dN[:, 2, 0] =  0.25 * yp * (2.0 * x + y)          # dN3/dξ
    dN[:, 2, 1] =  0.25 * xp * (2.0 * y + x)          # dN3/dη

    dN[:, 3, 0] =  0.25 * yp * (2.0 * x - y)          # dN4/dξ
    dN[:, 3, 1] =  0.25 * xm * (2.0 * y - x)          # dN4/dη

    dN[:, 4, 0] = -x * ym                              # dN5/dξ
    dN[:, 4, 1] = -0.5 * (one - x2)                   # dN5/dη

    dN[:, 5, 0] =  0.5 * (one - y2)                   # dN6/dξ
    dN[:, 5, 1] = -(one + x) * y                      # dN6/dη

    dN[:, 6, 0] = -x * yp                              # dN7/dξ
    dN[:, 6, 1] =  0.5 * (one - x2)                   # dN7/dη

    dN[:, 7, 0] = -0.5 * (one - y2)                   # dN8/dξ
    dN[:, 7, 1] = -(one - x) * y                      # dN8/dη

    return N, dN


def quad8_shape_functions_and_derivatives_all_ones(xi):
    """
    Buggy version for Q8 that returns all ones (no validation).
    Ensures consistent failure of partition of unity, derivative partition,
    Kronecker-delta, and completeness tests.
    """
    if isinstance(xi, np.ndarray) and xi.ndim == 1 and xi.shape == (2,):
        n = 1
    elif isinstance(xi, np.ndarray) and xi.ndim == 2 and xi.shape[1] == 2:
        n = xi.shape[0]
    else:
        xi = np.atleast_2d(xi)
        n = xi.shape[0]

    N = np.ones((n, 8, 1), dtype=float)
    dN = np.ones((n, 8, 2), dtype=float)
    return N, dN


def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_inputs = [
        [0.2, 0.3],                      # Python list, not np.ndarray
        np.array([[0.2, 0.3, 0.4]]),     # wrong shape (1,3)
        np.array([0.2, 0.3, 0.4]),       # wrong shape (3,)
        np.array([[0.2], [0.3]]),        # wrong shape (2,1)
        np.array([np.nan, 0.3]),         # contains NaN
        np.array([np.inf, 0.0]),         # contains Inf
    ]

    for bad in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad)


def test_partition_of_unity_quad8(fcn):
    """
    Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance.
    """
    pts = [
        (-1.0, -1.0),  # corner
        ( 1.0, -1.0),  # corner
        ( 1.0,  1.0),  # corner
        (-1.0,  1.0),  # corner
        ( 0.0, -1.0),  # mid-side bottom
        ( 1.0,  0.0),  # mid-side right
        ( 0.0,  1.0),  # mid-side top
        (-1.0,  0.0),  # mid-side left
        ( 0.0,  0.0),  # centroid
    ]

    for (xi, eta) in pts:
        N, _ = fcn(np.array([[xi, eta]], dtype=float))
        s = float(np.sum(N[0]))
        assert np.allclose(s, 1.0, atol=1e-12), f"Sum N != 1 at ({xi},{eta}); got {s}"


def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = [
        (-1.0, -1.0),  # corners
        ( 1.0, -1.0),
        ( 1.0,  1.0),
        (-1.0,  1.0),
        ( 0.0, -1.0),  # mid-sides
        ( 1.0,  0.0),
        ( 0.0,  1.0),
        (-1.0,  0.0),
        ( 0.0,  0.0),  # centroid
    ]

    for (xi, eta) in pts:
        _, dN = fcn(np.array([[xi, eta]], dtype=float))  # (1, 8, 2)
        grad_sum = np.sum(dN[0], axis=0)                # (2,)
        assert np.allclose(grad_sum, np.zeros(2), atol=1e-12), \
            f"Sum dN != 0 at ({xi},{eta}); got {grad_sum}"


def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([
        [-1.0, -1.0],  # Node 1 (corner)
        [ 1.0, -1.0],  # Node 2 (corner)
        [ 1.0,  1.0],  # Node 3 (corner)
        [-1.0,  1.0],  # Node 4 (corner)
        [ 0.0, -1.0],  # Node 5 (mid-side bottom)
        [ 1.0,  0.0],  # Node 6 (mid-side right)
        [ 0.0,  1.0],  # Node 7 (mid-side top)
        [-1.0,  0.0],  # Node 8 (mid-side left)
    ], dtype=float)

    M = np.zeros((8, 8), dtype=float)
    for j in range(8):
        N, _ = fcn(nodes[j].reshape(1, 2))  # evaluate at node j
        M[:, j] = N[0, :, 0]

    assert np.allclose(M, np.eye(8), atol=1e-12), f"Kronecker-delta matrix not identity:\n{M}"


def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    # Reference Q8 nodes: 4 corners + 4 mid-sides
    nodes = np.array([
        [-1.0, -1.0],  # 1
        [ 1.0, -1.0],  # 2
        [ 1.0,  1.0],  # 3
        [-1.0,  1.0],  # 4
        [ 0.0, -1.0],  # 5
        [ 1.0,  0.0],  # 6
        [ 0.0,  1.0],  # 7
        [-1.0,  0.0],  # 8
    ], dtype=float)

    # Sample points: corners, mid-sides, centroid
    pts = [
        (-1.0, -1.0), ( 1.0, -1.0), ( 1.0,  1.0), (-1.0,  1.0),  # corners
        ( 0.0, -1.0), ( 1.0,  0.0), ( 0.0,  1.0), (-1.0,  0.0),  # mid-sides
        ( 0.0,  0.0),                                             # centroid
    ]

    # Degree 1 polynomial: u = a + b ξ + c η
    def u1(xi, eta): return 0.25 - 0.7*xi + 1.3*eta
    u_nodes1 = np.array([u1(x, y) for (x, y) in nodes])
    max_err1 = 0.0
    for (xi, eta) in pts:
        N, _ = fcn(np.array([[xi, eta]], dtype=float))    # (1, 8, 1)
        u_h = float(N[0, :, 0] @ u_nodes1)
        max_err1 = max(max_err1, abs(u_h - u1(xi, eta)))
    assert max_err1 < 1e-12, f"Degree-1 completeness failed; max_err={max_err1:.3e}"

    # Degree 2 polynomial: u = a + b ξ + c η + d ξ² + e ξη + f η²
    def u2(xi, eta): return 0.1 - 0.2*xi + 0.3*eta + 0.4*xi*xi - 0.5*xi*eta + 0.6*eta*eta
    u_nodes2 = np.array([u2(x, y) for (x, y) in nodes])
    max_err2 = 0.0
    for (xi, eta) in pts:
        N, _ = fcn(np.array([[xi, eta]], dtype=float))    # (1, 8, 1)
        u_h = float(N[0, :, 0] @ u_nodes2)
        max_err2 = max(max_err2, abs(u_h - u2(xi, eta)))
    assert max_err2 < 1e-12, f"Degree-2 completeness failed; max_err={max_err2:.3e}"


def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    # Q8 reference nodes: 4 corners + 4 mid-sides
    nodes = np.array([
        [-1.0, -1.0],  # 1
        [ 1.0, -1.0],  # 2
        [ 1.0,  1.0],  # 3
        [-1.0,  1.0],  # 4
        [ 0.0, -1.0],  # 5
        [ 1.0,  0.0],  # 6
        [ 0.0,  1.0],  # 7
        [-1.0,  0.0],  # 8
    ], dtype=float)

    # Sample points: corners, mid-sides, centroid
    pts = [
        (-1.0, -1.0), ( 1.0, -1.0), ( 1.0,  1.0), (-1.0,  1.0),  # corners
        ( 0.0, -1.0), ( 1.0,  0.0), ( 0.0,  1.0), (-1.0,  0.0),  # mid-sides
        ( 0.0,  0.0),                                             # centroid
    ]

    # Degree 1: u = a + b ξ + c η  → ∇u = [b, c]
    def u1(xi, eta): return 0.25 - 0.7*xi + 1.3*eta
    def grad_u1(xi, eta): return np.array([-0.7, 1.3], dtype=float)
    u_nodes1 = np.array([u1(x, y) for (x, y) in nodes], dtype=float)
    max_err1 = 0.0
    for (xi, eta) in pts:
        _, dN = fcn(np.array([[xi, eta]], dtype=float))   # (1, 8, 2)
        grad_h = dN[0].T @ u_nodes1                      # (2,)
        max_err1 = max(max_err1, np.linalg.norm(grad_h - grad_u1(xi, eta)))
    assert max_err1 < 1e-12, f"Degree-1 gradient completeness failed; max_err={max_err1:.3e}"

    # Degree 2: u = a + b ξ + c η + d ξ² + e ξη + f η²
    def u2(xi, eta): return 0.1 - 0.2*xi + 0.3*eta + 0.4*xi*xi - 0.5*xi*eta + 0.6*eta*eta
    def grad_u2(xi, eta):
        return np.array([
            -0.2 + 0.8*xi - 0.5*eta,          # ∂u/∂ξ
             0.3 - 0.5*xi + 1.2*eta           # ∂u/∂η
        ], dtype=float)
    u_nodes2 = np.array([u2(x, y) for (x, y) in nodes], dtype=float)
    max_err2 = 0.0
    for (xi, eta) in pts:
        _, dN = fcn(np.array([[xi, eta]], dtype=float))   # (1, 8, 2)
        grad_h = dN[0].T @ u_nodes2                      # (2,)
        max_err2 = max(max_err2, np.linalg.norm(grad_h - grad_u2(xi, eta)))
    assert max_err2 < 1e-12, f"Degree-2 gradient completeness failed; max_err={max_err2:.3e}"


def task_info():
    task_id = "quad8_shape_functions_and_derivatives"
    task_short_description = "returns the values of shape functions and derivatives given natural coordinate points for an 8 node quad"
    created_date = "2025-09-23"
    created_by = "elejeune11"
    main_fcn = quad8_shape_functions_and_derivatives
    required_imports = ["import numpy as np", "import pytest", "from typing import Tuple"]
    fcn_dependencies = []
    reference_verification_inputs = [
        [np.array([0.2, -0.3])],            # interior single point
        [np.array([-1.0, -1.0])],           # corner
        [np.array([0.0, -1.0])],            # mid-side
        [np.array([0.0, 0.0])],             # centroid
        [np.array([[ -0.5, -0.2],
                   [  0.7,  0.1],
                   [  0.1,  0.8]])],        # batch of 3 points
    ]
    test_cases = [{"test_code": test_quad8_shape_functions_and_derivatives_input_errors, "expected_failures": [quad8_shape_functions_and_derivatives_no_error]},
                  {"test_code": test_partition_of_unity_quad8, "expected_failures": [quad8_shape_functions_and_derivatives_all_ones]},
                  {"test_code": test_derivative_partition_of_unity_quad8, "expected_failures": [quad8_shape_functions_and_derivatives_all_ones]},
                  {"test_code": test_kronecker_delta_at_nodes_quad8, "expected_failures": [quad8_shape_functions_and_derivatives_all_ones]},
                  {"test_code": test_value_completeness_quad8, "expected_failures": [quad8_shape_functions_and_derivatives_all_ones]},
                  {"test_code": test_gradient_completeness_quad8, "expected_failures": [quad8_shape_functions_and_derivatives_all_ones]}
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
