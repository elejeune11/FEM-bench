def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """

    def try_call(node_coords, elements):
        try:
            K = fcn(np.asarray(node_coords, dtype=float), elements)
            return (K, elements)
        except Exception:
            elements_1b = []
            for el in elements:
                el2 = dict(el)
                el2['node_i'] = int(el['node_i']) + 1
                el2['node_j'] = int(el['node_j']) + 1
                elements_1b.append(el2)
            K = fcn(np.asarray(node_coords, dtype=float), elements_1b)
            return (K, elements_1b)

    def check_result(node_coords, elements, K):
        n = len(node_coords)
        assert isinstance(K, np.ndarray)
        assert K.shape == (6 * n, 6 * n)
        assert np.allclose(K, K.T, atol=1e-08, rtol=1e-06)
        max_idx = max((int(el['node_i']) for el in elements)) if elements else -1
        if max_idx > n - 1:
            idx_offset = -1
        else:
            idx_offset = 0
        tol = 1e-12
        for el in elements:
            ni = int(el['node_i']) + idx_offset
            nj = int(el['node_j']) + idx_offset
            assert 0 <= ni < n and 0 <= nj < n
            rows = np.r_[6 * ni:6 * ni + 6, 6 * nj:6 * nj + 6]
            subK = K[np.ix_(rows, rows)]
            assert np.any(np.abs(subK) > tol)
    base_props = dict(E=210000000000.0, nu=0.3, A=0.01, I_y=1e-06, I_z=1e-06, J=1e-06)
    node_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    elements = [dict(node_i=0, node_j=1, **base_props)]
    (K, elems_used) = try_call(node_coords, elements)
    check_result(node_coords, elems_used, K)
    node_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    elements = [dict(node_i=0, node_j=1, **base_props), dict(node_i=1, node_j=2, **base_props)]
    (K, elems_used) = try_call(node_coords, elements)
    check_result(node_coords, elems_used, K)
    node_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.86602540378, 0.0)]
    elements = [dict(node_i=0, node_j=1, **base_props), dict(node_i=1, node_j=2, **base_props), dict(node_i=2, node_j=0, **base_props)]
    (K, elems_used) = try_call(node_coords, elements)
    check_result(node_coords, elems_used, K)
    node_coords = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    elements = [dict(node_i=0, node_j=1, **base_props), dict(node_i=1, node_j=2, **base_props), dict(node_i=2, node_j=3, **base_props), dict(node_i=3, node_j=0, **base_props)]
    (K, elems_used) = try_call(node_coords, elements)
    check_result(node_coords, elems_used, K)