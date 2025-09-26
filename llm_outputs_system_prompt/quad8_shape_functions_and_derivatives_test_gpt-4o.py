def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = [None, 5, 'string', np.array([1, 2, 3]), np.array([[1, 2], [3, 4], [5, 6]]), np.array([np.nan, 0]), np.array([np.inf, 0]), np.array([[0, 1], [np.nan, 0]]), np.array([[0, 1], [np.inf, 0]])]
    for bad_input in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad_input)

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(sample_points)
    gradient_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(gradient_sums, 0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    reference_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(reference_nodes)
    N_matrix = N[:, :, 0]
    identity_matrix = np.eye(8)
    assert np.allclose(N_matrix, identity_matrix, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""

    def polynomial(xi, eta):
        return 1 + xi + eta + xi ** 2 + eta ** 2 + xi * eta
    reference_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    nodal_values = polynomial(reference_nodes[:, 0], reference_nodes[:, 1])
    sample_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, 0]])
    (N, _) = fcn(sample_points)
    interpolated_values = np.einsum('ijk,j->ik', N, nodal_values).flatten()
    exact_values = polynomial(sample_points[:, 0], sample_points[:, 1])
    assert np.allclose(interpolated_values, exact_values, atol=1e-12)

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""

    def gradient(xi, eta):
        return np.array([1 + 2 * xi + eta, 1 + 2 * eta + xi])
    reference_nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    nodal_gradients = np.array([gradient(xi, eta) for (xi, eta) in reference_nodes])
    sample_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, 0]])
    (_, dN_dxi) = fcn(sample_points)
    interpolated_gradients = np.einsum('ijk,jk->ik', dN_dxi, nodal_gradients)
    exact_gradients = np.array([gradient(xi, eta) for (xi, eta) in sample_points])
    assert np.allclose(interpolated_gradients, exact_gradients, atol=1e-12)