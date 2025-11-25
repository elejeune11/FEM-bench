def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
or (n,2) with finite values. Invalid inputs should raise ValueError.
This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.1, 0.2])
    with pytest.raises(ValueError):
        fcn((0.1, 0.2))
    with pytest.raises(ValueError):
        fcn(np.array(0.5))
    with pytest.raises(ValueError):
        fcn(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2, 0.3]]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.1, 0.2]]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.1]))
    with pytest.raises(ValueError):
        fcn(np.array([0.1, np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2], [np.nan, 0.4]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
This test evaluates ∑ N_i at well considered sample points and ensures
that the sum equals 1 within tight tolerance."""
    xi_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.7], [0.4, 0.3]])
    (N, _) = fcn(xi_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
For every sample point, the vector sum equals (0,0) within tight tolerance."""
    xi_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.7], [0.4, 0.3]])
    (_, dN_dxi) = fcn(xi_points)
    sum_of_grads = np.sum(dN_dxi, axis=1)
    assert np.allclose(sum_of_grads, 0.0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each reference node location and assembles a 6×6 matrix whose
(i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    N_matrix = N.squeeze(axis=2)
    identity_matrix = np.eye(6)
    assert np.allclose(N_matrix, identity_matrix)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    coeffs = np.array([2.0, 3.0, -4.0, 5.0, -6.0, 7.0])

    def poly(xi, eta):
        return coeffs[0] + coeffs[1] * xi + coeffs[2] * eta + coeffs[3] * xi ** 2 + coeffs[4] * eta ** 2 + coeffs[5] * xi * eta
    nodal_values = poly(nodes[:, 0], nodes[:, 1])
    xi_points = np.array([[1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.7], [0.4, 0.3], [0.1, 0.1], [0.8, 0.1], [0.1, 0.8]])
    (N, _) = fcn(xi_points)
    interpolated_values = N.squeeze(axis=2) @ nodal_values
    exact_values = poly(xi_points[:, 0], xi_points[:, 1])
    assert np.allclose(interpolated_values, exact_values)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    coeffs = np.array([2.0, 3.0, -4.0, 5.0, -6.0, 7.0])

    def poly(xi, eta):
        return coeffs[0] + coeffs[1] * xi + coeffs[2] * eta + coeffs[3] * xi ** 2 + coeffs[4] * eta ** 2 + coeffs[5] * xi * eta

    def grad_poly(xi, eta):
        dp_dxi = coeffs[1] + 2 * coeffs[3] * xi + coeffs[5] * eta
        dp_deta = coeffs[2] + 2 * coeffs[4] * eta + coeffs[5] * xi
        return np.stack([dp_dxi, dp_deta], axis=-1)
    nodal_values = poly(nodes[:, 0], nodes[:, 1]).reshape(-1, 1)
    xi_points = np.array([[1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.7], [0.4, 0.3], [0.1, 0.1], [0.8, 0.1], [0.1, 0.8]])
    (_, dN_dxi) = fcn(xi_points)
    reconstructed_grad = np.sum(dN_dxi * nodal_values.reshape(1, 6, 1), axis=1)
    exact_grad = grad_poly(xi_points[:, 0], xi_points[:, 1])
    assert np.allclose(reconstructed_grad, exact_grad)