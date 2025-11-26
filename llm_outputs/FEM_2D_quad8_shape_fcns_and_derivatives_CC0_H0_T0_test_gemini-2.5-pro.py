def test_quad8_shape_functions_and_derivatives_input_errors(fcn: Callable):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
with finite values. Invalid inputs must raise ValueError. This test feeds a set of
bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = ['not an array', [0.0, 0.0], np.array([1, 2, 3]), np.zeros((5, 3)), np.zeros((2, 2, 2)), np.array([0.0, np.nan]), np.array([[0.0, 0.5], [np.inf, 0.2]])]
    for bad_input in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad_input)

def test_partition_of_unity_quad8(fcn: Callable):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
and centroid) and ensures that the sum equals 1 within tight tolerance."""
    xi_pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.123, -0.456]])
    (N, _) = fcn(xi_pts)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0)

def test_derivative_partition_of_unity_quad8(fcn: Callable):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
For every sample point, the vector sum equals (0,0) within tight tolerance."""
    xi_pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.123, -0.456]])
    (_, dN_dxi) = fcn(xi_pts)
    sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(sums, 0.0)

def test_kronecker_delta_at_nodes_quad8(fcn: Callable):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    N_matrix = N.squeeze(axis=-1)
    assert np.allclose(N_matrix, np.eye(8))

def test_value_completeness_quad8(fcn: Callable):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi_pts = np.array([[0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.123, -0.456], [-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    (xi, eta) = (xi_pts[:, 0], xi_pts[:, 1])
    (nodes_xi, nodes_eta) = (nodes[:, 0], nodes[:, 1])
    (N, _) = fcn(xi_pts)
    N_flat = N.squeeze(axis=-1)
    u_nodes_const = np.full(8, 5.0)
    u_exact_const = np.full(xi_pts.shape[0], 5.0)
    u_interp_const = N_flat @ u_nodes_const
    assert np.allclose(u_interp_const, u_exact_const)
    u_nodes_linear = 1.0 + 2.0 * nodes_xi + 3.0 * nodes_eta
    u_exact_linear = 1.0 + 2.0 * xi + 3.0 * eta
    u_interp_linear = N_flat @ u_nodes_linear
    assert np.allclose(u_interp_linear, u_exact_linear)
    u_nodes_quad = 1.0 + 2.0 * nodes_xi + 3.0 * nodes_eta + 4.0 * nodes_xi ** 2 + 5.0 * nodes_eta ** 2 + 6.0 * nodes_xi * nodes_eta
    u_exact_quad = 1.0 + 2.0 * xi + 3.0 * eta + 4.0 * xi ** 2 + 5.0 * eta ** 2 + 6.0 * xi * eta
    u_interp_quad = N_flat @ u_nodes_quad
    assert np.allclose(u_interp_quad, u_exact_quad)

def test_gradient_completeness_quad8(fcn: Callable):
    """Check that Q8 quad shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi_pts = np.array([[0.0, 0.0], [0.5, 0.5], [-0.25, 0.75], [0.123, -0.456], [-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    (xi, eta) = (xi_pts[:, 0], xi_pts[:, 1])
    (nodes_xi, nodes_eta) = (nodes[:, 0], nodes[:, 1])
    (_, dN_dxi) = fcn(xi_pts)
    u_nodes_const = np.full(8, 5.0)
    grad_u_exact_const = np.zeros_like(xi_pts)
    grad_u_interp_const = np.einsum('nij,j->ni', dN_dxi, u_nodes_const)
    assert np.allclose(grad_u_interp_const, grad_u_exact_const)
    u_nodes_linear = 1.0 + 2.0 * nodes_xi + 3.0 * nodes_eta
    grad_u_exact_linear = np.tile([2.0, 3.0], (xi_pts.shape[0], 1))
    grad_u_interp_linear = np.einsum('nij,j->ni', dN_dxi, u_nodes_linear)
    assert np.allclose(grad_u_interp_linear, grad_u_exact_linear)
    u_nodes_quad = 1.0 + 2.0 * nodes_xi + 3.0 * nodes_eta + 4.0 * nodes_xi ** 2 + 5.0 * nodes_eta ** 2 + 6.0 * nodes_xi * nodes_eta
    grad_u_exact_quad = np.zeros_like(xi_pts)
    grad_u_exact_quad[:, 0] = 2.0 + 8.0 * xi + 6.0 * eta
    grad_u_exact_quad[:, 1] = 3.0 + 10.0 * eta + 6.0 * xi
    grad_u_interp_quad = np.einsum('nij,j->ni', dN_dxi, u_nodes_quad)
    assert np.allclose(grad_u_interp_quad, grad_u_exact_quad)