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
    expected_x_coords = np.linspace(xl, xh, npx)
    expected_y_coords = np.linspace(yl, yh, npy)
    actual_x_coords = np.unique(coords1[:, 0])
    actual_y_coords = np.unique(coords1[:, 1])
    assert np.allclose(actual_x_coords, expected_x_coords)
    assert np.allclose(actual_y_coords, expected_y_coords)
    corner_nodes = [coords1[0], coords1[npx - 1], coords1[-npx], coords1[-1]]
    expected_corners = [[xl, yl], [xh, yl], [xl, yh], [xh, yh]]
    for (actual, expected) in zip(corner_nodes, expected_corners):
        assert np.allclose(actual, expected)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    for elem in connect:
        assert np.all(elem >= 0) and np.all(elem < n_nodes)
        assert len(np.unique(elem)) == 6
        corners = elem[:3]
        midsides = elem[3:]
        (n1, n2, n3) = corners
        (p1, p2, p3) = (coords[n1], coords[n2], coords[n3])
        area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        assert area > 0, 'Corners should be counter-clockwise'
        expected_m4 = 0.5 * (p1 + p2)
        expected_m5 = 0.5 * (p2 + p3)
        expected_m6 = 0.5 * (p3 + p1)
        assert np.allclose(coords[elem[3]], expected_m4)
        assert np.allclose(coords[elem[4]], expected_m5)
        assert np.allclose(coords[elem[5]], expected_m6)
    edge_dict = {}
    for (elem_idx, elem) in enumerate(connect):
        edges = [(elem[0], elem[1], elem[3]), (elem[1], elem[2], elem[4]), (elem[2], elem[0], elem[5])]
        for edge in edges:
            sorted_edge = tuple(sorted(edge[:2]))
            if sorted_edge in edge_dict:
                assert edge_dict[sorted_edge] == edge[2], 'Midside node mismatch on shared edge'
            else:
                edge_dict[sorted_edge] = edge[2]

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)