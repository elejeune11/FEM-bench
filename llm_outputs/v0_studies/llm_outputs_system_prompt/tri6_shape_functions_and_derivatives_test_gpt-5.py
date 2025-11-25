def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    np = fcn.__globals__['np']
    invalid_inputs = [0.5, 'not an array', [0.1, 0.2], (0.1, 0.2), np.array([0.1, 0.2, 0.3], dtype=float), np.array([[0.1], [0.2]], dtype=float), np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=float), np.array([np.nan, 0.0], dtype=float), np.array([[0.2, np.inf], [0.1, 0.2]], dtype=float)]
    for bad in invalid_inputs:
        raised = False
        try:
            fcn(bad)
        except ValueError:
            raised = True
        assert raised

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures
    that the sum equals 1 within tight tolerance.
    """
    np = fcn.__globals__['np']
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.6, 0.3]], dtype=float)
    (N, dN) = fcn(pts)
    s = np.sum(N[:, :, 0], axis=1)
    assert np.allclose(s, np.ones_like(s), rtol=0.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    np = fcn.__globals__['np']
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.6, 0.3]], dtype=float)
    (N, dN) = fcn(pts)
    grad_sum = np.sum(dN, axis=1)
    zeros = np.zeros_like(grad_sum)
    assert np.allclose(grad_sum, zeros, rtol=0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    np = fcn.__globals__['np']
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    (N, dN) = fcn(nodes)
    M = N[:, :, 0].T
    I = np.eye(6, dtype=float)
    assert np.allclose(M, I, rtol=0.0, atol=1e-13)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    np = fcn.__globals__['np']
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    samples = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.6, 0.3], [0.1, 0.7]], dtype=float)

    def p1(x, y):
        return 2.0 * x - 3.0 * y + 0.7

    def p2(x, y):
        return 1.1 + 2.2 * x - 3.3 * y + 4.4 * x * y + 5.5 * x * x - 6.6 * y * y
    for p in (p1, p2):
        v = np.array([p(x, y) for (x, y) in nodes], dtype=float)
        (N_s, _) = fcn(samples)
        u_h = N_s[:, :, 0] @ v
        u_exact = np.array([p(x, y) for (x, y) in samples], dtype=float)
        err = np.max(np.abs(u_h - u_exact))
        assert err <= 1e-13

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    np = fcn.__globals__['np']
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    samples = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.6, 0.3], [0.1, 0.7]], dtype=float)

    def p1(x, y):
        return 2.0 * x - 3.0 * y + 0.7

    def grad_p1(x, y):
        return np.array([2.0, -3.0], dtype=float)

    def p2(x, y):
        return 1.1 + 2.2 * x - 3.3 * y + 4.4 * x * y + 5.5 * x * x - 6.6 * y * y

    def grad_p2(x, y):
        return np.array([2.2 + 4.4 * y + 11.0 * x, -3.3 + 4.4 * x - 13.2 * y], dtype=float)
    for (p, grad_p) in ((p1, grad_p1), (p2, grad_p2)):
        v = np.array([p(x, y) for (x, y) in nodes], dtype=float)
        (_, dN_s) = fcn(samples)
        g_rec = dN_s.transpose(0, 2, 1) @ v
        g_exact = np.array([grad_p(x, y) for (x, y) in samples], dtype=float)
        err = np.max(np.linalg.norm(g_rec - g_exact, axis=1))
        assert err <= 1e-12