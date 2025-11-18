def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
or (n,2) with finite values. Invalid inputs should raise ValueError.
This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn(None)
    with pytest.raises(ValueError):
        fcn([0.1, 0.2])
    with pytest.raises(ValueError):
        fcn((0.1, 0.2))
    with pytest.raises(ValueError):
        fcn(np.array(0.1))
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
        fcn(np.array([[0.1, 0.2], [0.3, np.nan]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
This test evaluates ∑ N_i at well considered sample points and ensures
that the sum equals 1 within tight tolerance."""
    xi_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.7], [0.6, 0.2]])
    (N, _) = fcn(xi_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
For every sample point, the vector sum equals (0,0) within tight tolerance."""
    xi_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.7], [0.6, 0.2]])
    (_, dN_dxi) = fcn(xi_points)
    sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(sums, 0.0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each reference node location and assembles a 6×6 matrix whose
(i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    N_matrix = N.squeeze(axis=-1)
    assert np.allclose(N_matrix, np.eye(6))

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    xi_samples = np.array([[1 / 3, 1 / 3], [0.1, 0.1], [0.2, 0.5], [0.6, 0.2], [0.25, 0.0], [0.0, 0.75], [0.4, 0.4]])
    (N, _) = fcn(xi_samples)
    N_flat = N.squeeze(axis=-1)
    polys = [lambda x, y: 1.0 + 0 * x, lambda x, y: 2.0 - 3.0 * x + 4.0 * y, lambda x, y: 5.0 * x ** 2 - 6.0 * y ** 2 + 7.0 * x * y, lambda x, y: 1.0 + 2.0 * x - 3.0 * y + 4.0 * x ** 2 - 5.0 * y ** 2 + 6.0 * x * y]
    for p in polys:
        p_nodes = p(nodes[:, 0], nodes[:, 1])
        p_interp = N_flat @ p_nodes
        p_exact = p(xi_samples[:, 0], xi_samples[:, 1])
        assert np.allclose(p_interp, p_exact)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    xi_samples = np.array([[1 / 3, 1 / 3], [0.1, 0.1], [0.2, 0.5], [0.6, 0.2], [0.25, 0.0], [0.0, 0.75], [0.4, 0.4]])
    (_, dN_dxi) = fcn(xi_samples)
    poly_grads = [(lambda x, y: 1.0 + 0 * x, lambda x, y: np.array([0.0, 0.0])), (lambda x, y: 2.0 - 3.0 * x + 4.0 * y, lambda x, y: np.array([-3.0, 4.0])), (lambda x, y: 5.0 * x ** 2 - 6.0 * y ** 2 + 7.0 * x * y, lambda x, y: np.array([10.0 * x + 7.0 * y, -12.0 * y + 7.0 * x])), (lambda x, y: 1.0 + 2.0 * x - 3.0 * y + 4.0 * x ** 2 - 5.0 * y ** 2 + 6.0 * x * y, lambda x, y: np.array([2.0 + 8.0 * x + 6.0 * y, -3.0 - 10.0 * y + 6.0 * x]))]
    for (p, grad_p) in poly_grads:
        p_nodes = p(nodes[:, 0], nodes[:, 1])
        grad_p_interp = np.einsum('nji,j->ni', dN_dxi, p_nodes)
        grad_p_exact = np.zeros_like(grad_p_interp)
        for (i, (xi, eta)) in enumerate(xi_samples):
            grad_p_exact[i, :] = grad_p(xi, eta)
        assert np.allclose(grad_p_interp, grad_p_exact)