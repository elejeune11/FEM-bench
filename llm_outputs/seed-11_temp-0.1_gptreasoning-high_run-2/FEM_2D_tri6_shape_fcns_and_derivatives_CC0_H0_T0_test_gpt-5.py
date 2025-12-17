def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
or (n,2) with finite values. Invalid inputs should raise ValueError.
This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
that ValueError is raised.
    """
    with pytest.raises(ValueError):
        fcn([0.1, 0.2])
    with pytest.raises(ValueError):
        fcn((0.1, 0.2))
    with pytest.raises(ValueError):
        fcn(0.5)
    with pytest.raises(ValueError):
        fcn(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.ones((2, 1)))
    with pytest.raises(ValueError):
        fcn(np.ones((1, 3)))
    with pytest.raises(ValueError):
        fcn(np.ones((2, 2, 1)))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.2]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.2]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, -np.inf]]))

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
This test evaluates ∑ N_i at well considered sample points and ensures 
that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.7, 0.2], [0.25, 0.25]])
    N, dN = fcn(pts)
    assert N.shape == (pts.shape[0], 6, 1)
    assert dN.shape == (pts.shape[0], 6, 2)
    sums = N.sum(axis=1)[:, 0]
    assert np.allclose(sums, np.ones_like(sums), rtol=1e-13, atol=1e-13)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.7, 0.2], [0.25, 0.25]])
    _, dN = fcn(pts)
    grads_sum = dN.sum(axis=1)
    zeros = np.zeros_like(grads_sum)
    assert np.allclose(grads_sum, zeros, rtol=1e-13, atol=1e-13)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each reference node location and assembles a 6×6 matrix whose
(i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    N, _ = fcn(nodes)
    M = N[:, :, 0].T
    I = np.eye(6)
    assert M.shape == (6, 6)
    assert np.allclose(M, I, rtol=1e-13, atol=1e-13)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.7, 0.2], [0.25, 0.25]])

    def p1(x, y):
        return 3.0 + 2.0 * x - 0.5 * y

    def p2(x, y):
        return 1.0 + 0.5 * x + 0.75 * y + 1.2 * x * x - 0.3 * y * y + 0.8 * x * y
    for p in (p1, p2):
        phi_nodes = np.array([p(x, y) for x, y in nodes])
        N, _ = fcn(pts)
        interp = N[:, :, 0] @ phi_nodes
        exact = np.array([p(x, y) for x, y in pts])
        err = np.abs(interp - exact)
        assert np.max(err) < 1e-12

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.7, 0.2], [0.25, 0.25]])

    def p1(x, y):
        return 3.0 + 2.0 * x - 0.5 * y

    def grad_p1(x, y):
        return np.array([2.0, -0.5])

    def p2(x, y):
        return 1.0 + 0.5 * x + 0.75 * y + 1.2 * x * x - 0.3 * y * y + 0.8 * x * y

    def grad_p2(x, y):
        return np.array([0.5 + 2.4 * x + 0.8 * y, 0.75 - 0.6 * y + 0.8 * x])
    for p, grad_p in ((p1, grad_p1), (p2, grad_p2)):
        phi_nodes = np.array([p(x, y) for x, y in nodes])
        _, dN = fcn(pts)
        grads_interp = np.einsum('ijk,j->ik', dN, phi_nodes)
        grads_exact = np.array([grad_p(x, y) for x, y in pts])
        err = np.linalg.norm(grads_interp - grads_exact, axis=1)
        assert np.max(err) < 1e-12