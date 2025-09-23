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
    p2 = lambda x, y: x ** 2 + x * y + y ** 2 + x - y
    (N1, _) = fcn(xi)
    vals1 = np.array([p1(x, y) for (x, y) in zip(xi[:, 0], xi[:, 1])])
    vals2 = np.array([p2(x, y) for (x, y) in zip(xi[:, 0], xi[:, 1])])
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    nodal_vals1 = np.array([p1(x, y) for (x, y) in zip(nodes[:, 0], nodes[:, 1])]).reshape(-1, 1)
    nodal_vals2 = np.array([p2(x, y) for (x, y) in zip(nodes[:, 0], nodes[:, 1])]).reshape(-1, 1)
    interp1 = np.einsum('ij,jk->ik', N1, nodal_vals1)
    interp2 = np.einsum('ij,jk->ik', N1, nodal_vals2)
    assert np.allclose(interp1, vals1)
    assert np.allclose(interp2, vals2)

def test_gradient_completeness_tri6(fcn):
    xi = np.array([[0.2, 0.3], [0.7, 0.1], [0.1, 0.8]])
    p1 = lambda x, y: 2 * x + 3 * y - 1
    p2 = lambda x, y: x ** 2 + x * y + y ** 2 + x - y
    grad_p1 = lambda x, y: np.array([2, 3])
    grad_p2 = lambda x, y: np.array([2 * x + y + 1, x + 2 * y - 1])
    (_, dN_dxi) = fcn(xi)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    nodal_grads1 = np.array([grad_p1(x, y) for (x, y) in zip(nodes[:, 0], nodes[:, 1])])
    nodal_grads2 = np.array([grad_p2(x, y) for (x, y) in zip(nodes[:, 0], nodes[:, 1])])
    interp_grad1 = np.einsum('ijk,kl->ijl', dN_dxi, nodal_grads1)
    interp_grad2 = np.einsum('ijk,kl->ijl', dN_dxi, nodal_grads2)
    analytic_grad1 = np.array([grad_p1(x, y) for (x, y) in zip(xi[:, 0], xi[:, 1])])
    analytic_grad2 = np.array([grad_p2(x, y) for (x, y) in zip(xi[:, 0], xi[:, 1])])
    assert np.allclose(interp_grad1, analytic_grad1)
    assert np.allclose(interp_grad2, analytic_grad2)