def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    assert coords.shape[0] == expected_nodes
    assert coords.shape[1] == 2
    assert connect.shape[0] == nx * ny
    assert connect.shape[1] == 8
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.any(np.isclose(coords[:, 0], xl))
    assert np.any(np.isclose(coords[:, 0], xh))
    assert np.any(np.isclose(coords[:, 1], yl))
    assert np.any(np.isclose(coords[:, 1], yh))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    x_coords = np.unique(coords[:, 0])
    y_coords = np.unique(coords[:, 1])
    x_diffs = np.diff(x_coords)
    y_diffs = np.diff(y_coords)
    assert np.allclose(x_diffs, half_dx)
    assert np.allclose(y_diffs, half_dy)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0)
    assert np.all(connect < coords.shape[0])
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        corners = elem[:4]
        corner_coords = coords[corners]
        x = corner_coords[:, 0]
        y = corner_coords[:, 1]
        area = 0.5 * np.abs(x[0] * (y[1] - y[3]) + x[1] * (y[2] - y[0]) + x[2] * (y[3] - y[1]) + x[3] * (y[0] - y[2]))
        assert area > 0
    for elem in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem
        expected_n5 = 0.5 * (coords[n1] + coords[n2])
        assert np.allclose(coords[n5], expected_n5)
        expected_n6 = 0.5 * (coords[n2] + coords[n3])
        assert np.allclose(coords[n6], expected_n6)
        expected_n7 = 0.5 * (coords[n3] + coords[n4])
        assert np.allclose(coords[n7], expected_n7)
        expected_n8 = 0.5 * (coords[n4] + coords[n1])
        assert np.allclose(coords[n8], expected_n8)
    for cy in range(ny):
        for cx in range(nx - 1):
            elem_idx = cy * nx + cx
            next_elem_idx = elem_idx + 1
            right_nodes = [connect[elem_idx][1], connect[elem_idx][5], connect[elem_idx][2]]
            left_nodes = [connect[next_elem_idx][0], connect[next_elem_idx][7], connect[next_elem_idx][3]]
            assert connect[elem_idx][1] == connect[next_elem_idx][0]
            assert connect[elem_idx][2] == connect[next_elem_idx][3]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 2, 2)