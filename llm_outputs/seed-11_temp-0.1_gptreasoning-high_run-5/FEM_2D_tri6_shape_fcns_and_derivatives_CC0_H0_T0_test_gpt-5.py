def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,) or (n,2) with finite values. Invalid inputs should raise ValueError. This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts that ValueError is raised."""
    invalid_inputs = ['not-an-array', 3.14, [0.2, 0.3], np.array([0.1, 0.2, 0.3]), np.array([[0.1], [0.2]]), np.array([[0.1, 0.2, 0.3]]), np.array([np.nan, 0.2]), np.array([[0.1, np.inf]]), np.array([]), np.array([[0.1], [0.2], [0.3]])]
    for bad in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(bad)

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: sum_i N_i(xi,eta) = 1 for all points in the reference triangle. This test evaluates the sum at representative sample points and asserts it equals 1 within tight tolerance."""
    pts = []
    den = 4
    for i in range(den + 1):
        for j in range(den + 1 - i):
            pts.append([i / den, j / den])
    pts.extend([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3]])
    X = np.unique(np.array(pts, dtype=float), axis=0)
    N, dN = fcn(X)
    N2 = N[..., 0] if N.ndim == 3 else N
    s = np.sum(N2, axis=1)
    assert np.allclose(s, np.ones_like(s), rtol=0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies the sum of gradients equals zero: sum_i grad N_i = 0 for all sample points. This test verifies the vector sum equals (0,0) within tight tolerance at representative points."""
    pts = []
    den = 5
    for i in range(den + 1):
        for j in range(den + 1 - i):
            pts.append([i / den, j / den])
    X = np.unique(np.array(pts, dtype=float), axis=0)
    N, dN = fcn(X)
    dN_sum = np.sum(dN, axis=1)
    zeros = np.zeros_like(dN_sum)
    assert np.allclose(dN_sum, zeros, rtol=0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at node i and 0 at all other nodes. This test evaluates shape functions at each of the 6 reference node locations and checks the resulting 6x6 matrix equals the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    N, dN = fcn(nodes)
    N2 = N[..., 0] if N.ndim == 3 else N
    A = N2.T
    I = np.eye(6, dtype=float)
    assert np.allclose(A, I, rtol=0, atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials. Nodal values are set from the exact polynomial, the field is interpolated at sample points, and the maximum error is nearly zero."""
    pts = []
    den = 6
    for i in range(den + 1):
        for j in range(den + 1 - i):
            pts.append([i / den, j / den])
    X = np.array(pts, dtype=float)
    N, _ = fcn(X)
    N2 = N[..., 0] if N.ndim == 3 else N
    xi = X[:, 0]
    eta = X[:, 1]
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    xN = nodes[:, 0]
    eN = nodes[:, 1]
    polys = [(np.ones_like, np.ones_like), (lambda z: z[:, 0], lambda z: z[:, 0]), (lambda z: z[:, 1], lambda z: z[:, 1]), (lambda z: z[:, 0] ** 2, lambda z: z[:, 0] ** 2), (lambda z: z[:, 1] ** 2, lambda z: z[:, 1] ** 2), (lambda z: z[:, 0] * z[:, 1], lambda z: z[:, 0] * z[:, 1])]
    for idx, (pX_fn, pN_fn) in enumerate(polys):
        if idx == 0:
            p_vals = np.ones(X.shape[0], dtype=float)
            p_nodes = np.ones(nodes.shape[0], dtype=float)
        else:
            p_vals = pX_fn(X).astype(float)
            p_nodes = pN_fn(nodes).astype(float)
        interp = N2 @ p_nodes
        err = np.max(np.abs(interp - p_vals))
        assert err <= 1e-13

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal values and compared with the analytic gradient at sample points, with maximum error verified to be nearly zero."""
    pts = []
    den = 6
    for i in range(den + 1):
        for j in range(den + 1 - i):
            pts.append([i / den, j / den])
    X = np.array(pts, dtype=float)
    N, dN = fcn(X)
    xi = X[:, 0]
    eta = X[:, 1]
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=float)
    cases = [(lambda Z: np.ones(Z.shape[0], dtype=float), lambda X: np.column_stack([np.zeros(X.shape[0]), np.zeros(X.shape[0])])), (lambda Z: Z[:, 0], lambda X: np.column_stack([np.ones(X.shape[0]), np.zeros(X.shape[0])])), (lambda Z: Z[:, 1], lambda X: np.column_stack([np.zeros(X.shape[0]), np.ones(X.shape[0])])), (lambda Z: Z[:, 0] ** 2, lambda X: np.column_stack([2 * X[:, 0], np.zeros(X.shape[0])])), (lambda Z: Z[:, 1] ** 2, lambda X: np.column_stack([np.zeros(X.shape[0]), 2 * X[:, 1]])), (lambda Z: Z[:, 0] * Z[:, 1], lambda X: np.column_stack([X[:, 1], X[:, 0]]))]
    for nodal_fn, grad_fn in cases:
        a = nodal_fn(nodes).astype(float)
        grad_interp = np.einsum('nik,i->nk', dN, a)
        grad_exact = grad_fn(X).astype(float)
        err = np.max(np.abs(grad_interp - grad_exact))
        assert err <= 1e-13