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
        fcn(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.zeros((2, 1)))
    with pytest.raises(ValueError):
        fcn(np.zeros((1, 2, 2)))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2], [0.3, np.inf]]))

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
This test evaluates ∑ N_i at well considered sample points and ensures 
that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.4, 0.2]])
    (N, _) = fcn(pts)
    s = np.sum(N, axis=1)[:, 0]
    assert np.allclose(s, 1.0, atol=1e-13)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.4, 0.2]])
    (_, dN) = fcn(pts)
    grad_sum = np.sum(dN, axis=1)
    assert np.allclose(grad_sum, np.zeros_like(grad_sum), atol=1e-13)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
This test evaluates N at each reference node location and assembles a 6×6 matrix whose
(i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    A = N[:, :, 0]
    I = np.eye(6)
    assert np.allclose(A, I, atol=1e-13)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
degree-1 and degree-2 polynomials. Nodal values are set from the exact
polynomial, the field is interpolated at sample points, and the maximum
error is verified to be nearly zero.
    """
    pts = np.array([[0.1, 0.1], [0.25, 0.25], [0.6, 0.2], [0.2, 0.6 - 1e-12], [1.0 / 3.0, 1.0 / 3.0], [0.05, 0.7], [0.7, 0.05]])
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    polys = [lambda x, y: np.ones_like(x), lambda x, y: x, lambda x, y: y, lambda x, y: x ** 2, lambda x, y: y ** 2, lambda x, y: x * y]
    (N, _) = fcn(pts)
    N = N[:, :, 0]
    max_err = 0.0
    for poly in polys:
        nodal_vals = poly(nodes[:, 0], nodes[:, 1])
        interp_vals = N @ nodal_vals
        exact_vals = poly(pts[:, 0], pts[:, 1])
        err = np.max(np.abs(interp_vals - exact_vals))
        max_err = max(max_err, err)
    assert max_err < 1e-12

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
values and compared with the analytic gradient at sample points, with
maximum error verified to be nearly zero.
    """
    pts = np.array([[0.1, 0.1], [0.25, 0.25], [0.6, 0.2], [0.2, 0.6 - 1e-12], [1.0 / 3.0, 1.0 / 3.0], [0.05, 0.7], [0.7, 0.05]])
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])

    def f1(x, y):
        return np.ones_like(x)

    def g1(x, y):
        return (np.zeros_like(x), np.zeros_like(y))

    def f2(x, y):
        return x

    def g2(x, y):
        return (np.ones_like(x), np.zeros_like(y))

    def f3(x, y):
        return y

    def g3(x, y):
        return (np.zeros_like(x), np.ones_like(y))

    def f4(x, y):
        return x ** 2

    def g4(x, y):
        return (2.0 * x, np.zeros_like(y))

    def f5(x, y):
        return y ** 2

    def g5(x, y):
        return (np.zeros_like(x), 2.0 * y)

    def f6(x, y):
        return x * y

    def g6(x, y):
        return (y, x)
    polys = [(f1, g1), (f2, g2), (f3, g3), (f4, g4), (f5, g5), (f6, g6)]
    (_, dN) = fcn(pts)
    max_err = 0.0
    for (f, g) in polys:
        nodal_vals = f(nodes[:, 0], nodes[:, 1])
        grad_interp = np.einsum('nij,j->ni', dN, nodal_vals)
        (gx, gy) = g(pts[:, 0], pts[:, 1])
        grad_exact = np.column_stack([gx, gy])
        err = np.max(np.abs(grad_interp - grad_exact))
        max_err = max(max_err, err)
    assert max_err < 1e-12