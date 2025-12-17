def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn(42)
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1], [0.2]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2, 0.3]]))
    with pytest.raises(ValueError):
        fcn(np.array([]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3], [np.inf, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, np.nan], [0.1, 0.1]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    m = 5
    pts = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            pts.append([i / m, j / m])
    xi = np.array(pts, dtype=float)
    N, _ = fcn(xi)
    sumN = N[:, :, 0].sum(axis=1)
    assert np.allclose(sumN, np.ones_like(sumN), atol=1e-12, rtol=0.0)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    m = 5
    pts = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            pts.append([i / m, j / m])
    xi = np.array(pts, dtype=float)
    _, dN = fcn(xi)
    sum_dN = dN.sum(axis=1)
    assert np.allclose(sum_dN, np.zeros_like(sum_dN), atol=1e-12, rtol=0.0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    N, _ = fcn(nodes)
    M = N[:, :, 0]
    I = np.eye(6, dtype=float)
    assert np.allclose(M, I, atol=1e-12, rtol=0.0)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    m = 6
    pts = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            pts.append([i / m, j / m])
    xi = np.array(pts, dtype=float)

    def p_val(x, y, c):
        return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y
    coeffs_list = [np.array([2.3, -0.7, 1.1, 0.0, 0.0, 0.0], dtype=float), np.array([0.5, 1.2, -1.3, 0.7, -0.4, 0.9], dtype=float)]
    for c in coeffs_list:
        u_nodes = p_val(nodes[:, 0], nodes[:, 1], c)
        N, _ = fcn(xi)
        u_pred = (N[:, :, 0] * u_nodes[np.newaxis, :]).sum(axis=1)
        u_true = p_val(xi[:, 0], xi[:, 1], c)
        assert np.allclose(u_pred, u_true, atol=1e-12, rtol=0.0)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    m = 6
    pts = []
    for i in range(m + 1):
        for j in range(m + 1 - i):
            pts.append([i / m, j / m])
    xi = np.array(pts, dtype=float)

    def grad_val(x, y, c):
        gx = c[1] + 2.0 * c[3] * x + c[4] * y
        gy = c[2] + c[4] * x + 2.0 * c[5] * y
        return np.column_stack((gx, gy))
    coeffs_list = [np.array([2.3, -0.7, 1.1, 0.0, 0.0, 0.0], dtype=float), np.array([0.5, 1.2, -1.3, 0.7, -0.4, 0.9], dtype=float)]
    for c in coeffs_list:
        u_nodes = c[0] + c[1] * nodes[:, 0] + c[2] * nodes[:, 1] + c[3] * nodes[:, 0] ** 2 + c[4] * nodes[:, 0] * nodes[:, 1] + c[5] * nodes[:, 1] ** 2
        _, dN = fcn(xi)
        grad_pred = np.einsum('nsi,s->ni', dN, u_nodes)
        grad_true = grad_val(xi[:, 0], xi[:, 1], c)
        assert np.allclose(grad_pred, grad_true, atol=1e-12, rtol=0.0)