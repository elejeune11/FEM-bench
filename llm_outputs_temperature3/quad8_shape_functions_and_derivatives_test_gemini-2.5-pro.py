def test_quad8_shape_functions_and_derivatives_input_errors(fcn: Callable):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
with finite values. Invalid inputs must raise ValueError. This test feeds a set of
bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = [[0.0, 0.0], (0.0, 0.0), 'not an array', np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([[[0.0, 0.0]]]), np.array([np.nan, 0.0]), np.array([0.0, np.inf]), np.array([0.0, -np.inf]), np.array([[0.0, 0.0], [np.nan, 0.5]])]
    for bad_input in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad_input)

def test_partition_of_unity_quad8(fcn: Callable):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
and centroid) and ensures that the sum equals 1 within tight tolerance."""
    sample_xi = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.3, 0.7], [0.1, -0.9]])
    (N, _) = fcn(sample_xi)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0)

def test_derivative_partition_of_unity_quad8(fcn: Callable):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_xi = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.5, 0.5], [-0.3, 0.7], [0.1, -0.9]])
    (_, dN_dxi) = fcn(sample_xi)
    sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(sums, 0.0)

def test_kronecker_delta_at_nodes_quad8(fcn: Callable):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes_xi = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes_xi)
    N_at_nodes = N.squeeze().T
    identity_matrix = np.eye(8)
    assert np.allclose(N_at_nodes, identity_matrix)

def test_value_completeness_quad8(fcn: Callable):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero."""
    nodes_xi = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    grid = np.linspace(-1, 1, 5)
    (xx, yy) = np.meshgrid(grid, grid)
    sample_xi = np.vstack([xx.ravel(), yy.ravel()]).T

    def poly(xi, eta):
        return 2 - 3 * xi + 4 * eta + 5 * xi * eta - 6 * xi ** 2 + 7 * eta ** 2
    p_nodes = poly(nodes_xi[:, 0], nodes_xi[:, 1])
    (N_sample, _) = fcn(sample_xi)
    p_interp = np.einsum('nij,i->n', N_sample, p_nodes)
    p_exact = poly(sample_xi[:, 0], sample_xi[:, 1])
    assert np.allclose(p_interp, p_exact)

def test_gradient_completeness_quad8(fcn: Callable):
    """Check that Q8 quad shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero."""
    nodes_xi = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    grid = np.linspace(-1, 1, 5)
    (xx, yy) = np.meshgrid(grid, grid)
    sample_xi = np.vstack([xx.ravel(), yy.ravel()]).T

    def poly(xi, eta):
        return 2 - 3 * xi + 4 * eta + 5 * xi * eta - 6 * xi ** 2 + 7 * eta ** 2

    def grad_poly(xi, eta):
        dp_dxi = -3 + 5 * eta - 12 * xi
        dp_deta = 4 + 5 * xi + 14 * eta
        return np.vstack([dp_dxi, dp_deta]).T
    p_nodes = poly(nodes_xi[:, 0], nodes_xi[:, 1])
    (_, dN_dxi_sample) = fcn(sample_xi)
    grad_p_interp = np.einsum('nij,i->nj', dN_dxi_sample, p_nodes)
    grad_p_exact = grad_poly(sample_xi[:, 0], sample_xi[:, 1])
    assert np.allclose(grad_p_interp, grad_p_exact)