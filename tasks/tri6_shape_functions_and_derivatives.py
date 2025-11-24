import numpy as np
import pytest
from typing import Tuple


def tri6_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of quadratic (6-node) triangular shape functions and derivatives.

    Parameters
    ----------
    xi : np.ndarray
        Natural coordinates in the reference triangle.
        - Shape (2,) for a single point, or (n, 2) for a batch of points.
        - Must be finite (no NaN or Inf).

    Returns
    -------
    N : np.ndarray
        Shape functions evaluated at the input points. Shape: (n, 6, 1).
        Node order: [N1, N2, N3, N4, N5, N6].
    dN_dxi : np.ndarray
        Partial derivatives w.r.t (ξ, η). Shape: (n, 6, 2).
        Columns correspond to [∂()/∂ξ, ∂()/∂η] in the same node order.

    Raises
    ------
    ValueError
        If `xi` is not a NumPy array.
        If `xi` has shape other than (2,) or (n, 2).
        If `xi` contains non-finite values (NaN or Inf).

    Notes
    -----
    Uses P2 triangle with ξ_c = 1 - ξ - η:
        N1 = ξ(2ξ - 1),   N2 = η(2η - 1),   N3 = ξ_c(2ξ_c - 1),
        N4 = 4ξη,         N5 = 4ηξ_c,       N6 = 4ξξ_c.
    """
    if not isinstance(xi, np.ndarray):
        raise ValueError("xi must be a NumPy array.")
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError("1D xi must have shape (2,).")
        xi = xi.reshape(1, 2)  # promote to (n, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError("2D xi must have shape (n, 2).")
    else:
        raise ValueError("xi must have shape (2,) or (n, 2).")

    if not np.all(np.isfinite(xi)):
        raise ValueError("xi must contain finite values.")

    # Extract components
    xi1 = xi[:, 0].astype(float, copy=False)
    eta = xi[:, 1].astype(float, copy=False)
    xic = 1.0 - xi1 - eta

    n = xi.shape[0]

    # Allocate outputs
    N = np.empty((n, 6, 1), dtype=float)
    dN_dxi = np.empty((n, 6, 2), dtype=float)

    # Shape functions
    N[:, 0, 0] = (2.0 * xi1 - 1.0) * xi1            # N1
    N[:, 1, 0] = (2.0 * eta - 1.0) * eta            # N2
    N[:, 2, 0] = (2.0 * xic - 1.0) * xic            # N3
    N[:, 3, 0] = 4.0 * xi1 * eta                    # N4
    N[:, 4, 0] = 4.0 * eta * xic                    # N5
    N[:, 5, 0] = 4.0 * xic * xi1                    # N6

    # Derivatives: columns [d/dξ, d/dη]
    dN_dxi[:, 0, 0] = 4.0 * xi1 - 1.0               # ∂N1/∂ξ
    dN_dxi[:, 0, 1] = 0.0                            # ∂N1/∂η

    dN_dxi[:, 1, 0] = 0.0                            # ∂N2/∂ξ
    dN_dxi[:, 1, 1] = 4.0 * eta - 1.0                # ∂N2/∂η

    dN_dxi[:, 2, 0] = -(4.0 * xic - 1.0)             # ∂N3/∂ξ
    dN_dxi[:, 2, 1] = -(4.0 * xic - 1.0)             # ∂N3/∂η

    dN_dxi[:, 3, 0] = 4.0 * eta                      # ∂N4/∂ξ
    dN_dxi[:, 3, 1] = 4.0 * xi1                      # ∂N4/∂η

    dN_dxi[:, 4, 0] = -4.0 * eta                     # ∂N5/∂ξ
    dN_dxi[:, 4, 1] = 4.0 * (xic - eta)              # ∂N5/∂η

    dN_dxi[:, 5, 0] = 4.0 * (xic - xi1)              # ∂N6/∂ξ
    dN_dxi[:, 5, 1] = -4.0 * xi1                     # ∂N6/∂η

    return N, dN_dxi


def tri6_shape_functions_and_derivatives_no_error(xi):
    """
    Buggy version of D2_nn6_tri_with_grad_vec.

    This version does NOT validate inputs (wrong type, shape, or NaN values).
    As a result, tests that expect ValueError on invalid inputs will fail.
    """
    # Force input to at least 2D without validation
    xi = np.atleast_2d(xi).astype(float)
    xi1 = xi[:, 0]
    eta = xi[:, 1]
    xic = 1.0 - xi1 - eta

    n = xi.shape[0]
    N = np.empty((n, 6, 1), dtype=float)
    dN_dxi = np.empty((n, 6, 2), dtype=float)

    # Shape functions
    N[:, 0, 0] = (2.0 * xi1 - 1.0) * xi1
    N[:, 1, 0] = (2.0 * eta - 1.0) * eta
    N[:, 2, 0] = (2.0 * xic - 1.0) * xic
    N[:, 3, 0] = 4.0 * xi1 * eta
    N[:, 4, 0] = 4.0 * eta * xic
    N[:, 5, 0] = 4.0 * xic * xi1

    # Derivatives
    dN_dxi[:, 0, 0] = 4.0 * xi1 - 1.0
    dN_dxi[:, 0, 1] = 0.0
    dN_dxi[:, 1, 0] = 0.0
    dN_dxi[:, 1, 1] = 4.0 * eta - 1.0
    common = -(4.0 * xic - 1.0)
    dN_dxi[:, 2, 0] = common
    dN_dxi[:, 2, 1] = common
    dN_dxi[:, 3, 0] = 4.0 * eta
    dN_dxi[:, 3, 1] = 4.0 * xi1
    dN_dxi[:, 4, 0] = -4.0 * eta
    dN_dxi[:, 4, 1] = 4.0 * (xic - eta)
    dN_dxi[:, 5, 0] = 4.0 * (xic - xi1)
    dN_dxi[:, 5, 1] = -4.0 * xi1

    return N, dN_dxi


def tri6_shape_functions_and_derivatives_all_ones(xi):
    """
    Buggy version of D2_nn6_tri_with_grad_vec.

    Returns arrays filled with ones regardless of input.
    This guarantees consistent failure of tests for partition of unity,
    derivative partition, Kronecker-delta, and completeness.
    """
    if isinstance(xi, np.ndarray) and xi.ndim == 1 and xi.shape == (2,):
        n = 1
    elif isinstance(xi, np.ndarray) and xi.ndim == 2 and xi.shape[1] == 2:
        n = xi.shape[0]
    else:
        # For simplicity, force input into a (n,2) shape without raising
        xi = np.atleast_2d(xi)
        n = xi.shape[0]

    # Shape functions: (n, 6, 1)
    N = np.ones((n, 6, 1), dtype=float)

    # Derivatives: (n, 6, 2)
    dN = np.ones((n, 6, 2), dtype=float)

    return N, dN


def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    bad_inputs = [
        [0.2, 0.3],                      # Python list, not np.ndarray
        np.array([[0.2, 0.3, 0.4]]),     # wrong shape (1,3)
        np.array([0.2, 0.3, 0.4]),       # wrong shape (3,)
        np.array([[0.2], [0.3]]),        # wrong shape (2,1)
        np.array([np.nan, 0.3]),         # contains NaN
    ]

    for bad in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad)


def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance.
    """
    pts = [(1/3, 1/3)]
    a = 1/6
    pts += [(a, a), (1 - 2*a, a), (a, 1 - 2*a)]
    r15 = np.sqrt(15.0)
    b = (6 - r15) / 21.0
    g = (9 + 2*r15) / 21.0
    pts += [(b, b), (g, b), (b, g), (1/3, 1/3)]

    for (xi, eta) in pts:
        N, _ = fcn(np.array([[xi, eta]], dtype=float))
        s = float(np.sum(N[0]))
        assert np.allclose(s, 1.0, atol=1e-12), f"Sum N != 1 at ({xi},{eta}); got {s}"


def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = [(1/3, 1/3)]
    a = 1/6
    pts += [(a, a), (1 - 2*a, a), (a, 1 - 2*a)]
    r15 = np.sqrt(15.0)
    b = (6 - r15) / 21.0
    g = (9 + 2*r15) / 21.0
    pts += [(b, b), (g, b), (b, g), (1/3, 1/3)]

    for (xi, eta) in pts:
        _, dN = fcn(np.array([[xi, eta]], dtype=float))
        grad_sum = np.sum(dN[0], axis=0)
        assert np.allclose(grad_sum, np.zeros(2), atol=1e-12), f"Sum dN != 0 at ({xi},{eta}); got {grad_sum}"


def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [0.5, 0.0],
    ], dtype=float)

    M = np.zeros((6, 6), dtype=float)
    for j in range(6):
        N, _ = fcn(nodes[j].reshape(1, 2))
        M[:, j] = N[0, :, 0]

    assert np.allclose(M, np.eye(6), atol=1e-12), f"Kronecker-delta matrix not identity:\n{M}"


def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [0.5, 0.0],
    ], dtype=float)

    pts = [(1/3, 1/3)]
    a = 1/6
    pts += [(a, a), (1 - 2*a, a), (a, 1 - 2*a)]
    r15 = np.sqrt(15.0)
    b = (6 - r15) / 21.0
    g = (9 + 2*r15) / 21.0
    pts += [(b, b), (g, b), (b, g), (1/3, 1/3)]

    # Degree 1
    def u1(xi, eta): return 0.25 - 0.7*xi + 1.3*eta
    u_nodes1 = np.array([u1(x, y) for (x, y) in nodes])
    max_err1 = 0.0
    for (xi, eta) in pts:
        N, _ = fcn(np.array([[xi, eta]], dtype=float))
        u_h = float(N[0, :, 0] @ u_nodes1)
        max_err1 = max(max_err1, abs(u_h - u1(xi, eta)))
    assert max_err1 < 1e-12, f"Degree-1 completeness failed; max_err={max_err1:.3e}"

    # Degree 2
    def u2(xi, eta): return 0.1 - 0.2*xi + 0.3*eta + 0.4*xi*xi - 0.5*xi*eta + 0.6*eta*eta
    u_nodes2 = np.array([u2(x, y) for (x, y) in nodes])
    max_err2 = 0.0
    for (xi, eta) in pts:
        N, _ = fcn(np.array([[xi, eta]], dtype=float))
        u_h = float(N[0, :, 0] @ u_nodes2)
        max_err2 = max(max_err2, abs(u_h - u2(xi, eta)))
    assert max_err2 < 1e-12, f"Degree-2 completeness failed; max_err={max_err2:.3e}"


def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [0.5, 0.0],
    ], dtype=float)

    pts = [(1/3, 1/3)]
    a = 1/6
    pts += [(a, a), (1 - 2*a, a), (a, 1 - 2*a)]
    r15 = np.sqrt(15.0)
    b = (6 - r15) / 21.0
    g = (9 + 2*r15) / 21.0
    pts += [(b, b), (g, b), (b, g), (1/3, 1/3)]

    # Degree 1
    def u1(xi, eta): return 0.25 - 0.7*xi + 1.3*eta
    def grad_u1(xi, eta): return np.array([-0.7, 1.3])
    u_nodes1 = np.array([u1(x, y) for (x, y) in nodes])
    max_err1 = 0.0
    for (xi, eta) in pts:
        _, dN = fcn(np.array([[xi, eta]], dtype=float))
        grad_h = dN[0].T @ u_nodes1
        max_err1 = max(max_err1, np.linalg.norm(grad_h - grad_u1(xi, eta)))
    assert max_err1 < 1e-12, f"Degree-1 gradient completeness failed; max_err={max_err1:.3e}"

    # Degree 2
    def u2(xi, eta): return 0.1 - 0.2*xi + 0.3*eta + 0.4*xi*xi - 0.5*xi*eta + 0.6*eta*eta
    def grad_u2(xi, eta): return np.array([-0.2 + 0.8*xi - 0.5*eta,
                                           0.3 - 0.5*xi + 1.2*eta])
    u_nodes2 = np.array([u2(x, y) for (x, y) in nodes])
    max_err2 = 0.0
    for (xi, eta) in pts:
        _, dN = fcn(np.array([[xi, eta]], dtype=float))
        grad_h = dN[0].T @ u_nodes2
        max_err2 = max(max_err2, np.linalg.norm(grad_h - grad_u2(xi, eta)))
    assert max_err2 < 1e-12, f"Degree-2 gradient completeness failed; max_err={max_err2:.3e}"


def task_info():
    task_id = "tri6_shape_functions_and_derivatives"
    task_short_description = "returns the values of shape functions and derivatives given natural coordinate points for a 6 node triangle"
    created_date = "2025-09-23"
    created_by = "elejeune11"
    main_fcn = tri6_shape_functions_and_derivatives
    required_imports = ["import numpy as np", "import pytest", "from typing import Tuple"]
    fcn_dependencies = []
    reference_verification_inputs = [
        [np.array([0.2, 0.3])],             # interior single point
        [np.array([0.0, 0.0])],             # vertex
        [np.array([0.5, 0.0])],             # edge midpoint
        [np.array([1/3, 1/3])],             # centroid
        [np.array([[0.2, 0.2],
                   [0.1, 0.6],
                   [0.3, 0.1]])],           # batch of 3 points
    ]
    test_cases = [{"test_code": test_tri6_shape_functions_and_derivatives_input_errors, "expected_failures": [tri6_shape_functions_and_derivatives_no_error]},
                  {"test_code": test_partition_of_unity_tri6, "expected_failures": [tri6_shape_functions_and_derivatives_all_ones]},
                  {"test_code": test_derivative_partition_of_unity_tri6, "expected_failures": [tri6_shape_functions_and_derivatives_all_ones]},
                  {"test_code": test_kronecker_delta_at_nodes_tri6, "expected_failures": [tri6_shape_functions_and_derivatives_all_ones]},
                  {"test_code": test_value_completeness_tri6, "expected_failures": [tri6_shape_functions_and_derivatives_all_ones]},
                  {"test_code": test_gradient_completeness_tri6, "expected_failures": [tri6_shape_functions_and_derivatives_all_ones]}
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
