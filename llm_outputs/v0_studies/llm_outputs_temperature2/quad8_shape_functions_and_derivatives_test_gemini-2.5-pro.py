def test_quad8_shape_functions_and_derivatives_input_errors(fcn: Callable):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
with finite values. Invalid inputs must raise ValueError. This test feeds a set of
bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    with pytest.raises(ValueError):
        fcn(None)
    with pytest.raises(ValueError):
        fcn([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn('not an array')
    with pytest.raises(ValueError):
        fcn(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError):
        fcn(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([0.0, np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([0.0, -np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.0, 0.0], [np.nan, 0.0]]))

def test_partition_of_unity_quad8(fcn: Callable):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.123, -0.987]])
    (N_batch, _) = fcn(sample_points)
    sums_batch = np.sum(N_batch, axis=1)
    assert np.allclose(sums_batch, 1.0)
    (N_single, _) = fcn(np.array([0.2, -0.3]))
    sum_single = np.sum(N_single, axis=1)
    assert np.allclose(sum_single, 1.0)

def test_derivative_partition_of_unity_quad8(fcn: Callable):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.123, -0.987]])
    (_, dN_dxi_batch) = fcn(sample_points)
    sums_batch = np.sum(dN_dxi_batch, axis=1)
    assert np.allclose(sums_batch, 0.0)
    (_, dN_dxi_single) = fcn(np.array([0.2, -0.3]))
    sum_single = np.sum(dN_dxi_single, axis=1)
    assert np.allclose(sum_single, 0.0)

def test_kronecker_delta_at_nodes_quad8(fcn: Callable):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    N_matrix = N.squeeze().T
    identity_matrix = np.eye(8)
    assert np.allclose(N_matrix, identity_matrix)

def test_value_completeness_quad8(fcn: Callable):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    def poly(xi, eta):
        return 2 + 3 * xi - 4 * eta + 5 * xi * eta + 6 * xi ** 2 - 7 * eta ** 2 + 8 * xi ** 2 * eta - 9 * xi * eta ** 2
    u_nodes = poly(nodes[:, 0], nodes[:, 1])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.1, -0.9], [-0.8, -0.8], [0.7, -0.6]])
    (N, _) = fcn(sample_points)
    u_interp = N.squeeze() @ u_nodes
    u_exact = poly(sample_points[:, 0], sample_points[:, 1])
    assert np.allclose(u_interp, u_exact)

def test_gradient_completeness_quad8(fcn: Callable):
    """Check that Q8 quad shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    def poly(xi, eta):
        return 2 + 3 * xi - 4 * eta + 5 * xi * eta + 6 * xi ** 2 - 7 * eta ** 2 + 8 * xi ** 2 * eta - 9 * xi * eta ** 2

    def grad_poly(xi, eta):
        d_dxi = 3 + 5 * eta + 12 * xi + 16 * xi * eta - 9 * eta ** 2
        d_deta = -4 + 5 * xi - 14 * eta + 8 * xi ** 2 - 18 * xi * eta
        return np.vstack([d_dxi, d_deta]).T
    u_nodes = poly(nodes[:, 0], nodes[:, 1])
    sample_points = np.array([[0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.1, -0.9], [-0.8, -0.8], [0.7, -0.6]])
    (_, dN_dxi) = fcn(sample_points)
    grad_u_interp = np.einsum('i,jik->jk', u_nodes, dN_dxi)
    grad_u_exact = grad_poly(sample_points[:, 0], sample_points[:, 1])
    assert np.allclose(grad_u_interp, grad_u_exact)