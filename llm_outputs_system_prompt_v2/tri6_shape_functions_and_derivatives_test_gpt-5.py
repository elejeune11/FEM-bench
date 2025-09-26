def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn(0.5)
    with pytest.raises(ValueError):
        fcn(np.array([[0.1], [0.2]]))
    with pytest.raises(ValueError):
        fcn(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.1]))
    with pytest.raises(ValueError):
        fcn(np.array([0.1, np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3], [np.nan, 0.1]]))

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.3], [0.6, 0.2], [0.33, 0.33], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])
    (N, _) = fcn(pts)
    S = N.squeeze(-1).sum(axis=1)
    assert np.allclose(S, np.ones_like(S), atol=1e-13, rtol=0.0)
    (N_single, _) = fcn(np.array([0.25, 0.25]))
    S_single = float(N_single.squeeze(-1).sum())
    assert np.isclose(S_single, 1.0, atol=1e-13, rtol=0.0)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.3], [0.6, 0.2], [0.33, 0.33], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])
    (_, dN) = fcn(pts)
    grad_sum = dN.sum(axis=1)
    assert np.allclose(grad_sum, np.zeros_like(grad_sum), atol=1e-12, rtol=0.0)
    (_, dN_single) = fcn(np.array([0.25, 0.25]))
    grad_sum_single = dN_single.sum(axis=1)
    assert np.allclose(grad_sum_single, np.zeros_like(grad_sum_single), atol=1e-12, rtol=0.0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    A = np.zeros((6, 6))
    for j in range(6):
        (N, _) = fcn(nodes[j])
        A[:, j] = N.reshape(-1)
    I = np.eye(6)
    assert np.allclose(A, I, atol=1e-12, rtol=0.0)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.3], [0.6, 0.2], [0.3, 0.4], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])

    def eval_interp(vnod, x):
        (N, _) = fcn(x)
        return N.squeeze(-1) @ vnod

    def poly_list():

        def p1(xi, eta):
            return 3.0 - 2.0 * xi + 0.5 * eta

        def p2(xi, eta):
            return 5.0 + 2.0 * xi + 3.0 * eta + xi * eta + xi ** 2 - 0.75 * eta ** 2
        return [p1, p2]
    for p in poly_list():
        vnod = np.array([p(x, y) for (x, y) in nodes])
        u = eval_interp(vnod, pts)
        u_exact = np.array([p(x, y) for (x, y) in pts])
        assert np.allclose(u, u_exact, atol=1e-12, rtol=0.0)

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.3], [0.6, 0.2], [0.3, 0.4], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])

    def cases():

        def p1(xi, eta):
            return 3.0 - 2.0 * xi + 0.5 * eta

        def g1(xi, eta):
            return np.array([-2.0, 0.5])

        def p2(xi, eta):
            return 5.0 + 2.0 * xi + 3.0 * eta + xi * eta + xi ** 2 - 0.75 * eta ** 2

        def g2(xi, eta):
            dxi = 2.0 + eta + 2.0 * xi
            deta = 3.0 + xi - 1.5 * eta
            return np.array([dxi, deta])
        return [(p1, g1), (p2, g2)]
    for (p, g) in cases():
        vnod = np.array([p(x, y) for (x, y) in nodes])
        (_, dN) = fcn(pts)
        grad = np.einsum('n i a, i -> n a', dN, vnod)
        grad_exact = np.array([g(x, y) for (x, y) in pts])
        assert np.allclose(grad, grad_exact, atol=1e-12, rtol=0.0)