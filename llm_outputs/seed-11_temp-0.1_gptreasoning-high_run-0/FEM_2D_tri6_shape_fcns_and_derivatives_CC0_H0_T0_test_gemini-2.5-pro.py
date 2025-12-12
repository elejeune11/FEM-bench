def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
or (n,2) with finite values. Invalid inputs should raise ValueError.
This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.1, 0.2])
    with pytest.raises(ValueError):
        fcn(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2, 0.3]]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.1, 0.2]]]))
    with pytest.raises(ValueError):
        fcn(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2], [np.inf, 0.3]]))
    with pytest.raises(ValueError):
        fcn(np.array([0.1, -np.inf]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
This test evaluates ∑ N_i at well considered sample points and ensures
that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25], [0.1, 0.7]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25]])
    (_, dN_dxi) = fcn(sample_points)
    sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(sums, 0.0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each reference node location and assembles a 6×6 matrix whose
(i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    node_coords = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(node_coords)
    N_matrix = N.squeeze(axis=2)
    identity_matrix = np.eye(6)
    assert np.allclose(N_matrix, identity_matrix)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero."""
    node_coords = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    sample_points = np.array([[1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25], [0.1, 0.7], [0.6, 0.2], [0.2, 0.6], [0.5, 0.1]])
    coeffs = np.array([2.5, -1.2, 3.4, 0.5, -1.8, 2.1])

    def poly(xi_vec):
        (xi, eta) = (xi_vec[:, 0], xi_vec[:, 1])
        return coeffs[0] + coeffs[1] * xi + coeffs[2] * eta + coeffs[3] * xi ** 2 + coeffs[4] * eta ** 2 + coeffs[5] * xi * eta
    nodal_values = poly(node_coords)
    (N, _) = fcn(sample_points)
    N_flat = N.squeeze(axis=2)
    interpolated_values = N_flat @ nodal_values
    exact_values = poly(sample_points)
    assert np.allclose(interpolated_values, exact_values)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero."""
    node_coords = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    sample_points = np.array([[1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25], [0.1, 0.7], [0.6, 0.2], [0.2, 0.6], [0.5, 0.1]])
    coeffs = np.array([2.5, -1.2, 3.4, 0.5, -1.8, 2.1])

    def poly(xi_vec):
        (xi, eta) = (xi_vec[:, 0], xi_vec[:, 1])
        return coeffs[0] + coeffs[1] * xi + coeffs[2] * eta + coeffs[3] * xi ** 2 + coeffs[4] * eta ** 2 + coeffs[5] * xi * eta

    def grad_poly(xi_vec):
        (xi, eta) = (xi_vec[:, 0], xi_vec[:, 1])
        dp_dxi = coeffs[1] + 2 * coeffs[3] * xi + coeffs[5] * eta
        dp_deta = coeffs[2] + 2 * coeffs[4] * eta + coeffs[5] * xi
        return np.stack([dp_dxi, dp_deta], axis=-1)
    nodal_values = poly(node_coords)
    (_, dN_dxi) = fcn(sample_points)
    interpolated_grad = np.einsum('nji,j->ni', dN_dxi, nodal_values)
    exact_grad = grad_poly(sample_points)
    assert np.allclose(interpolated_grad, exact_grad)