def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """

    def dofs_for_node(i):
        return np.arange(6 * i, 6 * i + 6)
    props = dict(E=210000000000.0, nu=0.3, A=0.003, I_y=8e-06, I_z=5e-06, J=1.6e-05)
    atol = 1e-12
    rtol = 1e-12
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements = [{**props, 'node_i': 0, 'node_j': 1}]
    K = fcn(node_coords, elements)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, atol=atol, rtol=rtol)
    off_01 = K[np.ix_(dofs_for_node(0), dofs_for_node(1))]
    assert not np.allclose(off_01, 0.0, atol=atol)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{**props, 'node_i': 0, 'node_j': 1}, {**props, 'node_i': 1, 'node_j': 2}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T, atol=atol, rtol=rtol)
    b01 = K[np.ix_(dofs_for_node(0), dofs_for_node(1))]
    b12 = K[np.ix_(dofs_for_node(1), dofs_for_node(2))]
    b02 = K[np.ix_(dofs_for_node(0), dofs_for_node(2))]
    assert not np.allclose(b01, 0.0, atol=atol)
    assert not np.allclose(b12, 0.0, atol=atol)
    assert np.allclose(b02, 0.0, atol=atol)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]], dtype=float)
    elements = [{**props, 'node_i': 0, 'node_j': 1}, {**props, 'node_i': 1, 'node_j': 2}, {**props, 'node_i': 2, 'node_j': 0}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T, atol=atol, rtol=rtol)
    for (a, b) in [(0, 1), (1, 2), (2, 0)]:
        blk = K[np.ix_(dofs_for_node(a), dofs_for_node(b))]
        assert not np.allclose(blk, 0.0, atol=atol)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    elements = [{**props, 'node_i': 0, 'node_j': 1}, {**props, 'node_i': 1, 'node_j': 2}, {**props, 'node_i': 2, 'node_j': 3}, {**props, 'node_i': 3, 'node_j': 0}]
    K = fcn(node_coords, elements)
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T, atol=atol, rtol=rtol)
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        blk = K[np.ix_(dofs_for_node(a), dofs_for_node(b))]
        assert not np.allclose(blk, 0.0, atol=atol)
    for (a, b) in [(0, 2), (1, 3)]:
        blk = K[np.ix_(dofs_for_node(a), dofs_for_node(b))]
        assert np.allclose(blk, 0.0, atol=atol)