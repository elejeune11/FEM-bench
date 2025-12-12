def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    Test that FEM_2D_tri6_shape_fcns_and_derivatives_CC0_H0_T0 enforces input invariants.
    """
    with pytest.raises(ValueError):
        fcn('not a numpy array')
    with pytest.raises(ValueError):
        fcn(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        fcn(np.array([[1, 2], [3, 4], [5, 6]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 2]))
    with pytest.raises(ValueError):
        fcn(np.array([[1, 2], [3, np.nan]]))

def test_partition_of_unity_tri6(fcn):
    """
    Test that shape functions satisfy partition of unity.
    """
    xi = np.array([0.5, 0.25])
    (N, _) = fcn(xi)
    assert np.isclose(np.sum(N), 1.0)
    xi = np.array([[0.1, 0.2], [0.7, 0.1], [0.3, 0.6]])
    (N, _) = fcn(xi)
    assert np.allclose(np.sum(N, axis=1), np.ones(xi.shape[0]))

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Test that derivatives of shape functions sum to zero.
    """
    xi = np.array([0.5, 0.25])
    (_, dN_dxi) = fcn(xi)
    assert np.allclose(np.sum(dN_dxi, axis=1), np.zeros(2))
    xi = np.array([[0.1, 0.2], [0.7, 0.1], [0.3, 0.6]])
    (_, dN_dxi) = fcn(xi)
    assert np.allclose(np.sum(dN_dxi, axis=1), np.zeros((xi.shape[0], 2)))

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    Test that shape functions evaluate to Kronecker delta at nodes.
    """
    nodes = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5], [0, 0.5], [0.5, 0]])
    (N, _) = fcn(nodes)
    assert np.allclose(N, np.eye(6))

def test_value_completeness_tri6(fcn):
    """
    Test that shape functions can exactly reproduce degree-1 and degree-2 polynomials.
    """

    def p1(x, y):
        return x

    def p2(x, y):
        return x ** 2 + y ** 2
    nodes = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5], [0, 0.5], [0.5, 0]])
    p1_values = p1(nodes[:, 0], nodes[:, 1])
    p2_values = p2(nodes[:, 0], nodes[:, 1])
    xi = np.array([[0.1, 0.2], [0.7, 0.1], [0.3, 0.6]])
    (N, _) = fcn(xi)
    p1_interpolated = np.dot(N.reshape(-1, 6), p1_values)
    p2_interpolated = np.dot(N.reshape(-1, 6), p2_values)
    assert np.allclose(p1_interpolated, p1(xi[:, 0], xi[:, 1]))
    assert np.allclose(p2_interpolated, p2(xi[:, 0], xi[:, 1]))

def test_gradient_completeness_tri6(fcn):
    """
    Test that shape functions can exactly reproduce gradients of degree-1 and degree-2 polynomials.
    """

    def p1(x, y):
        return x

    def grad_p1(x, y):
        return np.array([1, 0])

    def p2(x, y):
        return x ** 2 + y ** 2

    def grad_p2(x, y):
        return np.array([2 * x, 2 * y])
    nodes = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5], [0, 0.5], [0.5, 0]])
    grad_p1_values = np.tile(grad_p1(0, 0), (6, 1))
    grad_p2_values = np.vstack([grad_p2(node[0], node[1]) for node in nodes])
    xi = np.array([[0.1, 0.2], [0.7, 0.1], [0.3, 0.6]])
    (_, dN_dxi) = fcn(xi)
    grad_p1_interpolated = np.dot(dN_dxi.reshape(-1, 6, 2), grad_p1_values)
    grad_p2_interpolated = np.dot(dN_dxi.reshape(-1, 6, 2), grad_p2_values)
    assert np.allclose(grad_p1_interpolated, np.tile(grad_p1(0, 0), (xi.shape[0], 1)))
    assert np.allclose(grad_p2_interpolated, np.array([grad_p2(x, y) for (x, y) in zip(xi[:, 0], xi[:, 1])]))