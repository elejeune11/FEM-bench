def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_nodes = npx * npy
    expected_elements = 2 * nx * ny
    assert coords1.shape == (expected_nodes, 2)
    assert connect1.shape == (expected_elements, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    corner_nodes = [coords1[0], coords1[npx - 1], coords1[-npx], coords1[-1]]
    expected_corners = [[xl, yl], [xh, yl], [xl, yh], [xh, yh]]
    for (actual, expected) in zip(corner_nodes, expected_corners):
        assert np.allclose(actual, expected)
    for i in range(npx):
        for j in range(npy):
            node_id = j * npx + i
            expected_x = xl + 0.5 * dx * i
            expected_y = yl + 0.5 * dy * j
            assert np.isclose(coords1[node_id, 0], expected_x)
            assert np.isclose(coords1[node_id, 1], expected_y)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    for elem in connect:
        assert np.all(elem >= 0) and np.all(elem < n_nodes)
        assert len(np.unique(elem)) == 6
        corners = elem[:3]
        v1 = coords[corners[1]] - coords[corners[0]]
        v2 = coords[corners[2]] - coords[corners[0]]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross_z > 0
        n4_expected = 0.5 * (coords[elem[0]] + coords[elem[1]])
        n5_expected = 0.5 * (coords[elem[1]] + coords[elem[2]])
        n6_expected = 0.5 * (coords[elem[2]] + coords[elem[0]])
        assert np.allclose(coords[elem[3]], n4_expected)
        assert np.allclose(coords[elem[4]], n5_expected)
        assert np.allclose(coords[elem[5]], n6_expected)
    npx = 2 * nx + 1
    edge_nodes = {}
    for elem in connect:
        edges = [(elem[0], elem[1], elem[3]), (elem[1], elem[2], elem[4]), (elem[2], elem[0], elem[5])]
        for (n1, n2, nmid) in edges:
            edge_key = tuple(sorted([n1, n2]))
            if edge_key in edge_nodes:
                assert edge_nodes[edge_key] == nmid
            else:
                edge_nodes[edge_key] = nmid

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0, 0, 1, 1, 0, 1)
    with pytest.raises(ValueError):
        fcn(0, 0, 1, 1, 1, 0)
    with pytest.raises(ValueError):
        fcn(1, 0, 0, 1, 1, 1)
    with pytest.raises(ValueError):
        fcn(0, 1, 1, 0, 1, 1)