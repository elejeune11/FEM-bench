def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    import numpy as np
    import pytest
    with pytest.raises(ValueError):
        fcn([[0, 0]])
    with pytest.raises(ValueError):
        fcn(np.array([0]))
    with pytest.raises(ValueError):
        fcn(np.array([0, 0, 0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0, 0, 0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0, np.nan]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    import numpy as np
    xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(xi)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    import numpy as np
    xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN) = fcn(xi)
    grad_sums = np.sum(dN, axis=1)
    assert np.allclose(grad_sums, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    N = N.squeeze()
    assert np.allclose(N, np.eye(8), atol=1e-14)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    xi = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0], [0, 0]])

    def f1(x, y):
        return 1 + 2 * x - 3 * y

    def f2(x, y):
        return 1 + 2 * x - 3 * y + 4 * x * y + 5 * x ** 2 + 6 * y ** 2
    for f in [f1, f2]:
        nodal_vals = f(nodes[:, 0], nodes[:, 1])
        (N, _) = fcn(xi)
        interp_vals = np.sum(N * nodal_vals, axis=1)
        exact_vals = f(xi[:, 0], xi[:, 1])
        assert np.allclose(interp_vals, exact_vals, atol=1e-14)

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    xi = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0], [0, 0]])

    def f1(x, y):
        return 1 + 2 * x - 3 * y

    def g1(x, y):
        return np.array([2 * np.ones_like(x), -3 * np.ones_like(y)]).T

    def f2(x, y):
        return 1 + 2 * x - 3 * y + 4 * x * y + 5 * x ** 2 + 6 * y ** 2

    def g2(x, y):
        return np.array([2 + 4 * y + 10 * x, -3 + 4 * x + 12 * y]).T
    for (f, g) in [(f1, g1), (f2, g2)]:
        nodal_vals = f(nodes[:, 0], nodes[:, 1])
        (_, dN) = fcn(xi)
        gra