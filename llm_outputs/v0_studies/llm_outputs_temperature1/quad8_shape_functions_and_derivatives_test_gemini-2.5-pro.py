def test_quad8_shape_functions_and_derivatives_input_errors(fcn: Callable):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = [[0.0, 0.0], (0.0, 0.0), 'not an array', None, np.array(5.0), np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0, 3.0]]), np.array([[[1.0, 2.0]]]), np.array([np.nan, 0.0]), np.array([0.0, np.inf]), np.array([-np.inf, 0.0]), np.array([[0.1, 0.2], [np.nan, 0.3]])]
    for bad_input in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad_input)

def test_partition_of_unity_quad8(fcn: Callable):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.123, -0.987]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0)

def test_derivative_partition_of_unity_quad8(fcn: Callable):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.123, -0.987]])
    (_, dN_dxi) = fcn(sample_points)
    sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(sums, 0.0)

def test_kronecker_delta_at_nodes_quad8(fcn: Callable):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    N_matrix = N.squeeze(axis=2)
    identity_matrix = np.eye(8)
    assert np.allclose(N_matrix, identity_matrix)

def test_value_completeness_quad8(fcn: Callable):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.1, -0.9], [-0.7, -0.3]])
    polynomials = [lambda xi, eta: 5.0, lambda xi, eta: 2.0 * xi - 3.0 * eta + 1.0, lambda xi, eta: xi * eta, lambda xi, eta: 4.0 * xi ** 2 - 2.0 * eta ** 2 + 3.0 * xi * eta - xi + 5.0 * eta - 1.0]
    (N_samples, _) = fcn(sample_points)
    N_samples = N_samples.squeeze(axis=2)
    for p in polynomials:
        u_nodes = p(nodes[:, 0], nodes[:, 1])
        u_interp = N_samples @ u_nodes
        u_exact = p(sample_points[:, 0], sample_points[:, 1])
        assert np.allclose(u_interp, u_exact)

def test_gradient_completeness_quad8(fcn: Callable):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.1, -0.9], [-0.7, -0.3]])
    fields = [(lambda xi, eta: 5.0, lambda xi, eta: np.array([0.0, 0.0])), (lambda xi, eta: 2.0 * xi - 3.0 * eta + 1.0, lambda xi, eta: np.array([2.0, -3.0])), (lambda xi, eta: xi * eta, lambda xi, eta: np.array([eta, xi])), (lambda xi, eta: 4.0 * xi ** 2 - 2.0 * eta ** 2 + 3.0 * xi * eta - xi + 5.0 * eta - 1.0, lambda xi, eta: np.array([8.0 * xi + 3.0 * eta - 1.0, -4.0 * eta + 3.0 * xi + 5.0]))]
    (_, dN_dxi_samples) = fcn(sample_points)
    for (p, grad_p) in fields:
        u_nodes = p(nodes[:, 0], nodes[:, 1])
        grad_interp = np.einsum('i,nji->nj', u_nodes, dN_dxi_samples)
        grad_exact = np.array([grad_p(xi, eta) for (xi, eta) in sample_points])
        assert np.allclose(grad_interp, grad_exact)