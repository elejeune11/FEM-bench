def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    import numpy as np
    import pytest
    with pytest.raises(ValueError):
        fcn([0, 0])
    with pytest.raises(ValueError):
        fcn(np.array([0]))
    with pytest.raises(ValueError):
        fcn(np.array([0, 0, 0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0, 0, 0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0]))
    with pytest.raises(ValueError):
        fcn(np.array([0, np.inf]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    import numpy as np
    xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(xi)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, rtol=1e-14, atol=1e-14)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    import numpy as np
    xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN) = fcn(xi)
    grad_sums = np.sum(dN, axis=1)
    assert np.allclose(grad_sums, 0.0, rtol=1e-14, atol=1e-14)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    N = N.squeeze()
    assert np.allclose(N, np.eye(8), rtol=1e-14, atol=1e-14)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    xi = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0], [0, 0]])
    polys = [lambda x, y: 1 + x + y, lambda x, y: 1 + x + y + x * y + x ** 2 + y ** 2]
    (N, _) = fcn(xi)
    for p in polys:
        nodal_vals = p(nodes[:, 0], nodes[:, 1])
        interp_vals = np.sum(N * nodal_vals, axis=1)
        exact_vals = p(xi[:, 0], xi[:, 1])
        assert np.allclose(interp_vals, exact_vals, rtol=1e-14, atol=1e-14)

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    xi = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0], [0, 0]])
    polys = [(lambda x, y: 1 + x + y, lambda x, y: np.array([np.ones_like(x), np.ones_like(y)]).T), (lambda x, y: 1 + x + y + x * y + x ** 2 + y ** 2, lambda x, y: np.array([1 + y + 2 * x, 1 + x + 2 * y]).T)]
    (_, dN) = fcn(xi)
    for (p, dp) in polys:
        nodal_vals = p(nodes[:, 0], nodes[:, 1])
        grad_vals = np.sum(dN * nodal_vals[:, None], axis=1)
        exact_grads = dp(xi[:, 0], xi[:, 1])
        assert np.allclose(grad_vals, exact_grads, rtol=1e-14, atol=1e-14)