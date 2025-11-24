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
        fcn(np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([-np.inf, 0.0]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5], [-0.5, -0.5], [0.3, -0.7]])
    (N, _) = fcn(test_points)
    sum_N = np.sum(N, axis=1)
    assert_allclose(sum_N, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], [0.5, 0.5], [-0.5, -0.5], [0.3, -0.7]])
    (_, dN_dxi) = fcn(test_points)
    sum_dN = np.sum(dN_dxi, axis=1)
    assert_allclose(sum_dN, 0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    N_matrix = N.reshape(8, 8)
    expected_identity = np.eye(8)
    assert_allclose(N_matrix, expected_identity, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    test_points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.7, 0.2], [0.8, -0.6]])
    linear_polys = [lambda xi, eta: 1.0 + 0 * xi, lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: 1.0 + 2 * xi - 3 * eta]
    quadratic_polys = [lambda xi, eta: xi * eta, lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: 1.0 + 2 * xi - 3 * eta + 0.5 * xi * eta + 1.5 * xi ** 2 - 2 * eta ** 2]
    all_polys = linear_polys + quadratic_polys
    for poly in all_polys:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (N_test, _) = fcn(test_points)
        N_test = N_test.reshape(len(test_points), 8)
        interpolated = N_test @ nodal_values
        exact = np.array([poly(pt[0], pt[1]) for pt in test_points])
        max_error = np.max(np.abs(interpolated - exact))
        assert max_error < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    test_points = np.array([[0.0, 0.0], [0.5, 0.3], [-0.7, 0.2], [0.8, -0.6]])
    linear_cases = [{'func': lambda xi, eta: 1.0 + 0 * xi, 'grad': lambda xi, eta: np.array([0.0, 0.0])}, {'func': lambda xi, eta: xi, 'grad': lambda xi, eta: np.array([1.0, 0.0])}, {'func': lambda xi, eta: eta, 'grad': lambda xi, eta: np.array([0.0, 1.0])}, {'func': lambda xi, eta: 1.0 + 2 * xi - 3 * eta, 'grad': lambda xi, eta: np.array([2.0, -3.0])}]
    quadratic_cases = [{'func': lambda xi, eta: xi * eta, 'grad': lambda xi, eta: np.array([eta, xi])}, {'func': lambda xi, eta: xi ** 2, 'grad': lambda xi, eta: np.array([2 * xi, 0.0])}, {'func': lambda xi, eta: eta ** 2, 'grad': lambda xi, eta: np.array([0.0, 2 * eta])}, {'func': lambda xi, eta: 1.0 + 2 * xi - 3 * eta + 0.5 * xi * eta + 1.5 * xi ** 2 - 2 * eta ** 2, 'grad': lambda xi, eta: np.array([2.0 + 0.5 * eta + 3 * xi, -3.0 + 0.5 * xi - 4 * eta])}]
    all_cases = linear_cases + quadratic_cases
    for case in all_cases:
        poly = case['func']
        grad_exact_func = case['grad']
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        (N_test, dN_dxi_test) = fcn(test_points)
        N_test = N_test.reshape(len(test_points), 8)
        dN_dxi_test = dN_dxi_test.reshape(len(test_points), 8, 2)
        grad_reconstructed = np.zeros((len(test_points), 2))
        for i in range(8):
            grad_reconstructed[:, 0] += nodal_values[i] * dN_dxi_test[:, i, 0]
            grad_reconstructed[:, 1] += nodal_values[i] * dN_dxi_test[:, i, 1]
        grad_exact = np.array([grad_exact_func(pt[0], pt[1]) for pt in test_points])
        max_error = np.max(np.abs(grad_reconstructed - grad_exact))
        assert max_error < 1e-12