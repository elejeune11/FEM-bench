def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    The test covers multiple structural configurations:
    """

    def block(K, i, j):
        return K[6 * i:6 * i + 6, 6 * j:6 * j + 6]

    def assert_sym_shape(K, n_nodes):
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.all(np.isfinite(K))
        assert np.allclose(K, K.T, rtol=1e-09, atol=1e-09)

    def assert_nonzero_block(K, i, j, tol=1e-12):
        b = block(K, i, j)
        assert np.any(np.abs(b) > tol)

    def assert_zero_block(K, i, j, tol=1e-12):
        b = block(K, i, j)
        assert np.all(np.abs(b) <= tol)

    def assert_element_12x12_block_nonzero(K, i, j, tol=1e-12):
        rows = np.r_[np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)]
        blk = K[np.ix_(rows, rows)]
        assert np.any(np.abs(blk) > tol)
    material = dict(E=210000000000.0, nu=0.3, A=0.01, I_y=1e-06, I_z=1e-06, J=2e-06)
    z_up = [0.0, 0.0, 1.0]
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [dict(node_i=0, node_j=1, local_z=z_up, **material)]
    K = fcn(nodes, elements)
    assert_sym_shape(K, 2)
    assert_nonzero_block(K, 0, 0)
    assert_nonzero_block(K, 1, 1)
    assert_nonzero_block(K, 0, 1)
    assert_nonzero_block(K, 1, 0)
    assert_element_12x12_block_nonzero(K, 0, 1)
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [dict(node_i=0, node_j=1, local_z=z_up, **material), dict(node_i=1, node_j=2, local_z=z_up, **material)]
    K = fcn(nodes, elements)
    assert_sym_shape(K, 3)
    for i in range(3):
        assert_nonzero_block(K, i, i)
    assert_nonzero_block(K, 0, 1)
    assert_nonzero_block(K, 1, 0)
    assert_nonzero_block(K, 1, 2)
    assert_nonzero_block(K, 2, 1)
    assert_zero_block(K, 0, 2)
    assert_zero_block(K, 2, 0)
    for e in elements:
        assert_element_12x12_block_nonzero(K, e['node_i'], e['node_j'])
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2.0, 0.0]])
    elements = [dict(node_i=0, node_j=1, local_z=z_up, **material), dict(node_i=1, node_j=2, local_z=z_up, **material), dict(node_i=2, node_j=0, local_z=z_up, **material)]
    K = fcn(nodes, elements)
    assert_sym_shape(K, 3)
    for i in range(3):
        assert_nonzero_block(K, i, i)
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        assert_nonzero_block(K, i, j)
        assert_nonzero_block(K, j, i)
    for e in elements:
        assert_element_12x12_block_nonzero(K, e['node_i'], e['node_j'])
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements = [dict(node_i=0, node_j=1, local_z=z_up, **material), dict(node_i=1, node_j=2, local_z=z_up, **material), dict(node_i=2, node_j=3, local_z=z_up, **material), dict(node_i=3, node_j=0, local_z=z_up, **material)]
    K = fcn(nodes, elements)
    assert_sym_shape(K, 4)
    for i in range(4):
        assert_nonzero_block(K, i, i)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        assert_nonzero_block(K, i, j)
        assert_nonzero_block(K, j, i)
    for i, j in [(0, 2), (1, 3)]:
        assert_zero_block(K, i, j)
        assert_zero_block(K, j, i)
    for e in elements:
        assert_element_12x12_block_nonzero(K, e['node_i'], e['node_j'])