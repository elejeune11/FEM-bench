def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn((0.0, 0.0))
    with pytest.raises(ValueError):
        fcn(np.array([0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([0.0, np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([-np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0], [1.0, np.nan]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5], [-0.5, 0.3], [0.7, -0.2]])
    (N, _) = fcn(test_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5], [-0.5, 0.3], [0.7, -0.2]])
    (_, dN_dxi) = fcn(test_points)
    gradient_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(gradient_sums, 0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    N_matrix = N.reshape(8, 8)
    expected_identity = np.eye(8)
    assert np.allclose(N_matrix, expected_identity, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    test_points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.7, 0.2], [0.1, -0.8], [-0.9, -0.4]])
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N_test, _) = fcn(test_points)
    (N_nodes, _) = fcn(nodes)
    coefficients = [1.0, 2.5, -1.3]
    for (a, b, c) in [coefficients]:
        f_exact_nodes = a + b * nodes[:, 0] + c * nodes[:, 1]
        f_interp = np.sum(N_test[:, :, 0] * f_exact_nodes, axis=1)
        f_exact_test = a + b * test_points[:, 0] + c * test_points[:, 1]
        assert np.max(np.abs(f_interp - f_exact_test)) < 1e-12
    quadratic_coeffs = [1.0, 0.5, -0.8, 1.2, -0.9, 0.7]
    (a, b, c, d, e, f) = quadratic_coeffs
    f_exact_nodes = a + b * nodes[:, 0] + c * nodes[:, 1] + d * nodes[:, 0] ** 2 + e * nodes[:, 1] ** 2 + f * nodes[:, 0] * nodes[:, 1]
    f_interp = np.sum(N_test[:, :, 0] * f_exact_nodes, axis=1)
    f_exact_test = a + b * test_points[:, 0] + c * test_points[:, 1] + d * test_points[:, 0] ** 2 + e * test_points[:, 1] ** 2 + f * test_points[:, 0] * test_points[:, 1]
    assert np.max(np.abs(f_interp - f_exact_test)) < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    test_points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.7, 0.2], [0.1, -0.8], [-0.9, -0.4]])
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (_, dN_test) = fcn(test_points)
    (N_nodes, _) = fcn(nodes)
    coefficients = [1.0, 2.5, -1.3]
    for (a, b, c) in [coefficients]:
        f_exact_nodes = a + b * nodes[:, 0] + c * nodes[:, 1]
        grad_interp = np.zeros_like(test_points)
        for i in range(8):
            grad_interp[:, 0] += f_exact_nodes[i] * dN_test[:, i, 0]
            grad_interp[:, 1] += f_exact_nodes[i] * dN_test[:, i, 1]
        grad_exact = np.array([b, c])
        assert np.max(np.abs(grad_interp - grad_exact)) < 1e-12
    quadratic_coeffs = [1.0, 0.5, -0.8, 1.2, -0.9, 0.7]
    (a, b, c, d, e, f_val) = quadratic_coeffs
    f_exact_nodes = a + b * nodes[:, 0] + c * nodes[:, 1] + d * nodes[:, 0] ** 2 + e * nodes[:, 1] ** 2 + f_val * nodes[:, 0] * nodes[:, 1]
    grad_interp = np.zeros_like(test_points)
    for i in range(8):
        grad_interp[:, 0] += f_exact_nodes[i] * dN_test[:, i, 0]
        grad_interp[:, 1] += f_exact_nodes[i] * dN_test[:, i, 1]
    grad_exact = np.zeros_like(test_points)
    grad_exact[:, 0] = b + 2 * d * test_points[:, 0] + f_val * test_points[:, 1]
    grad_exact[:, 1] = c + 2 * e * test_points[:, 1] + f_val * test_points[:, 0]
    assert np.max(np.abs(grad_interp - grad_exact)) < 1e-12