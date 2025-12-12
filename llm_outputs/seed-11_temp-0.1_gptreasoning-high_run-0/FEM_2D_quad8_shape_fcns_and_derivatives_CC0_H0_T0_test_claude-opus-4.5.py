def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn((0.0, 0.0))
    with pytest.raises(ValueError):
        fcn(np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.0, 0.0]]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([0.0, -np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0], [np.nan, 0.0]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.5, 0.3]])
    (N, _) = fcn(sample_points)
    N_sum = np.sum(N, axis=1).flatten()
    assert np.allclose(N_sum, 1.0, atol=1e-12), f'Partition of unity failed: {N_sum}'

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.3, 0.7]])
    (_, dN_dxi) = fcn(sample_points)
    dN_sum = np.sum(dN_dxi, axis=1)
    assert np.allclose(dN_sum, 0.0, atol=1e-12), f'Derivative partition of unity failed: {dN_sum}'

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    N_matrix = N[:, :, 0]
    identity = np.eye(8)
    assert np.allclose(N_matrix, identity, atol=1e-12), f'Kronecker delta property failed:\n{N_matrix}'

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.3, -0.7], [-0.2, 0.1]])
    polynomials = [lambda xi, eta: np.ones_like(xi), lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: xi * eta, lambda xi, eta: 1 + 2 * xi - 3 * eta + xi ** 2 + 0.5 * eta ** 2 - xi * eta]
    (N_sample, _) = fcn(sample_points)
    for poly in polynomials:
        nodal_values = poly(nodes[:, 0], nodes[:, 1])
        interpolated = np.sum(N_sample[:, :, 0] * nodal_values, axis=1)
        exact = poly(sample_points[:, 0], sample_points[:, 1])
        max_error = np.max(np.abs(interpolated - exact))
        assert max_error < 1e-12, f'Value completeness failed with max error {max_error}'

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.3, -0.7], [-0.2, 0.1]])
    test_cases = [(lambda xi, eta: xi, lambda xi, eta: np.ones_like(xi), lambda xi, eta: np.zeros_like(xi)), (lambda xi, eta: eta, lambda xi, eta: np.zeros_like(xi), lambda xi, eta: np.ones_like(xi)), (lambda xi, eta: xi ** 2, lambda xi, eta: 2 * xi, lambda xi, eta: np.zeros_like(xi)), (lambda xi, eta: eta ** 2, lambda xi, eta: np.zeros_like(xi), lambda xi, eta: 2 * eta), (lambda xi, eta: xi * eta, lambda xi, eta: eta, lambda xi, eta: xi), (lambda xi, eta: 1 + 2 * xi - 3 * eta + xi ** 2 + 0.5 * eta ** 2 - xi * eta, lambda xi, eta: 2 + 2 * xi - eta, lambda xi, eta: -3 + eta - xi)]
    (_, dN_dxi_sample) = fcn(sample_points)
    for (poly, grad_xi, grad_eta) in test_cases:
        nodal_values = poly(nodes[:, 0], nodes[:, 1])
        interp_grad_xi = np.sum(dN_dxi_sample[:, :, 0] * nodal_values, axis=1)
        interp_grad_eta = np.sum(dN_dxi_sample[:, :, 1] * nodal_values, axis=1)
        exact_grad_xi = grad_xi(sample_points[:, 0], sample_points[:, 1])
        exact_grad_eta = grad_eta(sample_points[:, 0], sample_points[:, 1])
        max_error_xi = np.max(np.abs(interp_grad_xi - exact_grad_xi))
        max_error_eta = np.max(np.abs(interp_grad_eta - exact_grad_eta))
        assert max_error_xi < 1e-12, f'Gradient completeness (xi) failed with max error {max_error_xi}'
        assert max_error_eta < 1e-12, f'Gradient completeness (eta) failed with max error {max_error_eta}'