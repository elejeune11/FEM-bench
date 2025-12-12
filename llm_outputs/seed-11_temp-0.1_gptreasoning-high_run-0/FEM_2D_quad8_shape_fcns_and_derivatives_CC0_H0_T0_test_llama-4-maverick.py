def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        fcn(np.array([[1, 2], [3, 4, 5]]))
    with pytest.raises(ValueError):
        fcn([1, 2])
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 2]))
    with pytest.raises(ValueError):
        fcn(np.array([1, np.inf]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(sample_points)
    assert np.allclose(np.sum(N, axis=1), np.ones((sample_points.shape[0], 1)))

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(sample_points)
    assert np.allclose(np.sum(dN_dxi, axis=1), np.zeros((sample_points.shape[0], 2)))

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    N_matrix = np.squeeze(N)
    assert np.allclose(N_matrix, np.eye(8))

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    sample_points = np.random.uniform(-1, 1, (100, 2))
    for degree in [1, 2]:
        if degree == 1:
            u_exact = lambda x: x[:, 0] + 2 * x[:, 1]
        else:
            u_exact = lambda x: x[:, 0] ** 2 + 2 * x[:, 1] ** 2 + x[:, 0] * x[:, 1]
        u_nodes = u_exact(nodes)
        (N, _) = fcn(sample_points)
        u_interpolated = np.sum(N * u_nodes[:, np.newaxis], axis=1)
        assert np.allclose(u_interpolated, u_exact(sample_points), atol=1e-10)

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    sample_points = np.random.uniform(-1, 1, (100, 2))
    for degree in [1, 2]:
        if degree == 1:
            u_exact = lambda x: x[:, 0] + 2 * x[:, 1]
            grad_u_exact = lambda x: np.stack([np.ones_like(x[:, 0]), 2 * np.ones_like(x[:, 0])], axis=-1)
        else:
            u_exact = lambda x: x[:, 0] ** 2 + 2 * x[:, 1] ** 2 + x[:, 0] * x[:, 1]
            grad_u_exact = lambda x: np.stack([2 * x[:, 0] + x[:, 1], 4 * x[:, 1] + x[:, 0]], axis=-1)
        u_nodes = u_exact(nodes)
        (_, dN_dxi) = fcn(sample_points)
        grad_u_interpolated = np.sum(dN_dxi * u_nodes[:, np.newaxis], axis=1)
        assert np.allclose(grad_u_interpolated, grad_u_exact(sample_points), atol=1e-10)