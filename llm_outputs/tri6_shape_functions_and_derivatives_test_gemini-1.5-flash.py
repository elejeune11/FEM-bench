def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    with pytest.raises(ValueError):
        fcn(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        fcn([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 2]))
    with pytest.raises(ValueError):
        fcn(np.array([1, np.inf]))
    with pytest.raises(ValueError):
        fcn([1, 2])

def test_partition_of_unity_tri6(fcn):
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    (N, _) = fcn(xi)
    assert np.allclose(np.sum(N, axis=1), 1.0)

def test_derivative_partition_of_unity_tri6(fcn):
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    (_, dN_dxi) = fcn(xi)
    assert np.allclose(np.sum(dN_dxi, axis=1), 0.0)

def test_kronecker_delta_at_nodes_tri6(fcn):
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    (N, _) = fcn(nodes)
    assert np.allclose(N, np.identity(6))

def test_value_completeness_tri6(fcn):
    xi = np.array([[0.2, 0.3], [0.7, 0.1], [0.1, 0.8]])
    p1 = lambda x, y: 2 * x + 3 * y - 1
    p2 = lambda x, y: x ** 2 + x * y + y ** 2
    (N1, _) = fcn(xi)
    p1_nodes = np.array([p1(x, y) for (x, y) in [[0, 0], [1, 0], [0, 1], [0.5, 0], [0, 0.5], [0.5, 0.5]]]).reshape(-1, 1)
    p1_interp = np.dot(N1.T, p1_nodes)
    p1_exact = np.array([p1(x, y) for (x, y) in xi]).reshape(-1, 1)
    assert np.allclose(p1_interp, p1_exact)
    (N2, _) = fcn(xi)
    p2_nodes = np.array([p2(x, y) for (x, y) in [[0, 0], [1, 0], [0, 1], [0.5, 0], [0, 0.5], [0.5, 0.5]]]).reshape(-1, 1)
    p2_interp = np.dot(N2.T, p2_nodes)
    p2_exact = np.array([p2(x, y) for (x, y) in xi]).reshape(-1, 1)
    assert np.allclose(p2_interp, p2_exact)

def test_gradient_completeness_tri6(fcn):
    xi = np.array([[0.2, 0.3], [0.7, 0.1], [0.1, 0.8]])
    p1 = lambda x, y: 2 * x + 3 * y - 1
    p2 = lambda x, y: x ** 2 + x * y + y ** 2
    (_, dN1_dxi) = fcn(xi)
    grad_p1_nodes = np.array([[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]])
    grad_p1_interp = np.dot(dN1_dxi.T, grad_p1_nodes)
    grad_p1_exact = np.array([[2, 3], [2, 3], [2, 3]])
    assert np.allclose(grad_p1_interp, grad_p1_exact)
    (_, dN2_dxi) = fcn(xi)
    grad_p2_nodes = np.array([[0, 0], [2 * 1 + 0.5, 0.5], [0.5, 2 * 1 + 0.5], [1, 0.5], [0.5, 1], [1, 1]])
    grad_p2_interp = np.dot(dN2_dxi.T, grad_p2_nodes)
    grad_p2_exact = np.array([[2 * 0.2 + 0.3, 0.2 + 2 * 0.3], [2 * 0.7 + 0.1, 0.7 + 2 * 0.1], [2 * 0.1 + 0.8, 0.1 + 2 * 0.8]])
    assert np.allclose(grad_p2_interp, grad_p2_exact)