def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn((0.0, 0.0))
    with pytest.raises(ValueError):
        fcn(np.array([0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0], [0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, np.nan], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, -np.inf], [0.0, 0.0]]))

def test_partition_of_unity_quad8(fcn):
    """
    Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance.
    """
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]])
    (N, _) = fcn(sample_points)
    N_sum = np.sum(N[:, :, 0], axis=1)
    np.testing.assert_allclose(N_sum, np.ones(len(sample_points)), atol=1e-12, rtol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]])
    (_, dN_dxi) = fcn(sample_points)
    dN_sum = np.sum(dN_dxi, axis=1)
    np.testing.assert_allclose(dN_sum, np.zeros((len(sample_points), 2)), atol=1e-12, rtol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    N_matrix = N[:, :, 0].T
    np.testing.assert_allclose(N_matrix, np.eye(8), atol=1e-12, rtol=1e-12)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.3], [0.2, -0.7], [-0.8, 0.6]])

    def poly_deg1(xi, eta):
        return 1.0 + 2.0 * xi + 3.0 * eta
    nodal_values_deg1 = poly_deg1(nodes[:, 0], nodes[:, 1])
    (N, _) = fcn(sample_points)
    interpolated_deg1 = np.sum(N[:, :, 0] * nodal_values_deg1[np.newaxis, :], axis=1)
    exact_deg1 = poly_deg1(sample_points[:, 0], sample_points[:, 1])
    error_deg1 = np.max(np.abs(interpolated_deg1 - exact_deg1))
    assert error_deg1 < 1e-10

    def poly_deg2(xi, eta):
        return 1.0 + xi + eta + xi ** 2 + eta ** 2 + xi * eta
    nodal_values_deg2 = poly_deg2(nodes[:, 0], nodes[:, 1])
    interpolated_deg2 = np.sum(N[:, :, 0] * nodal_values_deg2[np.newaxis, :], axis=1)
    exact_deg2 = poly_deg2(sample_points[:, 0], sample_points[:, 1])
    error_deg2 = np.max(np.abs(interpolated_deg2 - exact_deg2))
    assert error_deg2 < 1e-10

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.3], [0.2, -0.7], [-0.8, 0.6]])

    def poly_deg1(xi, eta):
        return 1.0 + 2.0 * xi + 3.0 * eta

    def grad_poly_deg1(xi, eta):
        return np.array([2.0, 3.0])
    nodal_values_deg1 = poly_deg1(nodes[:, 0], nodes[:, 1])
    (_, dN_dxi) = fcn(sample_points)
    reconstructed_grad_deg1 = np.sum(dN_dxi * nodal_values_deg1[np.newaxis, :, np.newaxis], axis=1)
    exact_grad_deg1 = np.array([grad_poly_deg1(xi, eta) for (xi, eta) in sample_points])
    error_grad_deg1 = np.max(np.abs(reconstructed_grad_deg1 - exact_grad_deg1))
    assert error_grad_deg1 < 1e-10

    def poly_deg2(xi, eta):
        return 1.0 + xi + eta + xi ** 2 + eta ** 2 + xi * eta

    def grad_poly_deg2(xi, eta):
        return np.array([1.0 + 2.0 * xi + eta, 1.0 + 2.0 * eta + xi])
    nodal_values_deg2 = poly_deg2(nodes[:, 0], nodes[:, 1])
    reconstructed_grad_deg2 = np.sum(dN_dxi * nodal_values_deg2[np.newaxis, :, np.newaxis], axis=1)
    exact_grad_deg2 = np.array([grad_poly_deg2(xi, eta) for (xi, eta) in sample_points])
    error_grad_deg2 = np.max(np.abs(reconstructed_grad_deg2 - exact_grad_deg2))
    assert error_grad_deg2 < 1e-10