def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_nelems = nx * ny
    assert coords.shape[0] == expected_nnodes
    assert connect.shape[0] == expected_nelems
    assert coords.shape == (expected_nnodes, 2)
    assert connect.shape == (expected_nelems, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    assert np.any(np.isclose(x_coords, xl))
    assert np.any(np.isclose(x_coords, xh))
    assert np.any(np.isclose(y_coords, yl))
    assert np.any(np.isclose(y_coords, yh))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_x = np.array([xl + 0.5 * dx * i for i in range(2 * nx + 1)])
    expected_y = np.array([yl + 0.5 * dy * i for i in range(2 * ny + 1)])
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    expected_x_filtered = []
    expected_y_filtered = []
    for iy in range(2 * ny + 1):
        for ix in range(2 * nx + 1):
            if not (ix % 2 == 1 and iy % 2 == 1):
                if xl + 0.5 * dx * ix not in expected_x_filtered:
                    expected_x_filtered.append(xl + 0.5 * dx * ix)
                if yl + 0.5 * dy * iy not in expected_y_filtered:
                    expected_y_filtered.append(yl + 0.5 * dy * iy)
    expected_x_filtered = np.unique(expected_x_filtered)
    expected_y_filtered = np.unique(expected_y_filtered)
    assert np.allclose(np.sort(unique_x), np.sort(expected_x_filtered))
    assert np.allclose(np.sort(unique_y), np.sort(expected_y_filtered))
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 0.5, 2.0, 3.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0)
    assert np.all(connect < coords.shape[0])
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        corners = coords[elem[:4]]
        x = corners[:, 0]
        y = corners[:, 1]
        area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])
        assert area > 0
    for elem in connect:
        (n1, n2, n3, n4) = elem[:4]
        (n5, n6, n7, n8) = elem[4:]
        expected_n5 = 0.5 * (coords[n1] + coords[n2])
        assert np.allclose(coords[n5], expected_n5)
        expected_n6 = 0.5 * (coords[n2] + coords[n3])
        assert np.allclose(coords[n6], expected_n6)
        expected_n7 = 0.5 * (coords[n3] + coords[n4])
        assert np.allclose(coords[n7], expected_n7)
        expected_n8 = 0.5 * (coords[n4] + coords[n1])
        assert np.allclose(coords[n8], expected_n8)
    edge_nodes = {}
    for (elem_idx, elem) in enumerate(connect):
        (n1, n2, n3, n4) = elem[:4]
        (n5, n6, n7, n8) = elem[4:]
        edge_key = tuple(sorted([n1, n2]))
        if edge_key in edge_nodes:
            assert edge_nodes[edge_key] == n5
        else:
            edge_nodes[edge_key] = n5
        edge_key = tuple(sorted([n2, n3]))
        if edge_key in edge_nodes:
            assert edge_nodes[edge_key] == n6
        else:
            edge_nodes[edge_key] = n6
        edge_key = tuple(sorted([n3, n4]))
        if edge_key in edge_nodes:
            assert edge_nodes[edge_key] == n7
        else:
            edge_nodes[edge_key] = n7
        edge_key = tuple(sorted([n4, n1]))
        if edge_key in edge_nodes:
            assert edge_nodes[edge_key] == n8
        else:
            edge_nodes[edge_key] = n8

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)