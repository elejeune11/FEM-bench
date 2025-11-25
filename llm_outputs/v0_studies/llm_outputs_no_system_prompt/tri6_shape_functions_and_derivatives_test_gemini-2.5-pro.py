def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    invalid_inputs = [[0.5, 0.5], (0.5, 0.5), 'not an array', np.array([0.1, 0.2, 0.3]), np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), np.array([[[0.1, 0.2]]]), np.array([0.1, np.nan]), np.array([[0.1, 0.2], [np.inf, 0.3]])]
    for xi in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(xi)

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures
    that the sum equals 1 within tight tolerance.
    """
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25], [0.1, 0.7], [0.4, 0.3]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    expected_sums = np.ones((sample_points.shape[0], 1))
    assert_allclose(sums, expected_sums, atol=1e-15)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25]])
    (_, dN_dxi) = fcn(sample_points)
    sums = np.sum(dN_dxi, axis=1)
    expected_sums = np.zeros((sample_points.shape[0], 2))
    assert_allclose(sums, expected_sums, atol=1e-15)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    node_coords = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(node_coords)
    N_matrix = np.squeeze(N, axis=2).T
    expected_matrix = np.eye(6)
    assert_allclose(N_matrix, expected_matrix, atol=1e-15)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    node_coords = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    sample_points = np.array([[1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25], [0.1, 0.7], [0.4, 0.3], [0.1, 0.1], [0.8, 0.1], [0.1, 0.8]])
    (xi, eta) = (sample_points[:, 0], sample_points[:, 1])
    polynomials = [lambda xi, eta: 5.0, lambda xi, eta: 2 * xi - 3 * eta + 1, lambda xi, eta: 3 * xi ** 2 - 2 * eta ** 2 + 5 * xi * eta - xi + 2 * eta + 4]
    (N, _) = fcn(sample_points)
    for p in polynomials:
        u_nodes = p(node_coords[:, 0], node_coords[:, 1]).reshape(6, 1)
        u_interp = np.squeeze(N.transpose(0, 2, 1) @ u_nodes)
        u_exact = p(xi, eta)
        assert_allclose(u_interp, u_exact, atol=1e-14)

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    node_coords = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    sample_points = np.array([[1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25], [0.1, 0.7], [0.4, 0.3], [0.1, 0.1], [0.8, 0.1], [0.1, 0.8]])
    (xi, eta) = (sample_points[:, 0], sample_points[:, 1])
    test_cases = [(lambda xi, eta: 5.0, lambda xi, eta: np.zeros((xi.shape[0], 2))), (lambda xi, eta: 2 * xi - 3 * eta + 1, lambda xi, eta: np.full((xi.shape[0], 2), [2.0, -3.0])), (lambda xi, eta: 3 * xi ** 2 - 2 * eta ** 2 + 5 * xi * eta - xi + 2 * eta + 4, lambda xi, eta: np.vstack([6 * xi + 5 * eta - 1, -4 * eta + 5 * xi + 2]).T)]
    (_, dN_dxi) = fcn(sample_points)
    for (p, grad_p) in test_cases:
        u_nodes = p(node_coords[:, 0], node_coords[:, 1]).reshape(6, 1)
        grad_u_interp = np.squeeze(dN_dxi.transpose(0, 2, 1) @ u_nodes, axis=2)
        grad_u_exact = grad_p(xi, eta)
        assert_allclose(grad_u_interp, grad_u_exact, atol=1e-14)