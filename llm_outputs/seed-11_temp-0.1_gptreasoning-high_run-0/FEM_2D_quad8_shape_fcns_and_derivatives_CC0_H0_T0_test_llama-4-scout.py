def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    with pytest.raises(ValueError):
        fcn('not a numpy array')
    with pytest.raises(ValueError):
        fcn(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        fcn(np.array([[1, 2], [3, 4]], dtype=object))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 2]))

def test_partition_of_unity_quad8(fcn):
    """
    Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance.
    """
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN_dxi) = fcn(sample_points)
    sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(sums, np.zeros((sums.shape[0], 2)), atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(node_coords)
    kronecker_matrix = N.reshape((8, 8))
    assert np.allclose(kronecker_matrix, np.eye(8), atol=1e-12)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    x = lambda xi, eta: xi + eta
    nodal_values = x(*node_coords.T)
    (N, _) = fcn(node_coords)
    interpolated_values = np.dot(N.squeeze(), nodal_values)
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(sample_points)
    exact_values = x(*sample_points.T)
    interpolated = np.dot(N, nodal_values)
    assert np.allclose(interpolated, exact_values, atol=1e-12)
    x = lambda xi, eta: xi ** 2 + eta ** 2
    nodal_values = x(*node_coords.T)
    (N, _) = fcn(node_coords)
    interpolated_values = np.dot(N.squeeze(), nodal_values)
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(sample_points)
    exact_values = x(*sample_points.T)
    interpolated = np.dot(N, nodal_values)
    assert np.allclose(interpolated, exact_values, atol=1e-12)

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    x = lambda xi, eta: xi + eta
    nodal_values = x(*node_coords.T)
    (_, dN_dxi) = fcn(node_coords)
    grad_x_exact = np.array([1, 1])
    grad_reconstructed = np.dot(dN_dxi.squeeze().T, nodal_values)
    assert np.allclose(grad_reconstructed, grad_x_exact, atol=1e-12)
    x = lambda xi, eta: xi ** 2 + eta ** 2
    nodal_values = x(*node_coords.T)
    (_, dN_dxi) = fcn(node_coords)
    grad_x_exact = lambda xi, eta: np.array([2 * xi, 2 * eta])
    sample_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    exact_grads = grad_x_exact(*sample_points.T)
    (N, dN_dxi) = fcn(sample_points)
    reconstructed_grads = np.dot(dN_dxi, nodal_values)
    assert np.allclose(reconstructed_grads, exact_grads, atol=1e-12)