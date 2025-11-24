def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_elements = nx * ny
    assert coords1.shape == (expected_nodes, 2)
    assert connect1.shape == (expected_elements, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    corner_coords = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    for corner in corner_coords:
        assert np.any(np.all(np.isclose(coords1, corner), axis=1))
    for coord in coords1:
        x_rem = (coord[0] - xl) % (0.5 * dx)
        y_rem = (coord[1] - yl) % (0.5 * dy)
        assert np.isclose(x_rem, 0.0) or np.isclose(x_rem, 0.5 * dx)
        assert np.isclose(y_rem, 0.0) or np.isclose(y_rem, 0.5 * dy)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    for elem in connect:
        assert np.all(elem >= 0) and np.all(elem < n_nodes)
        assert len(np.unique(elem)) == 8
        corners = elem[:4]
        corner_coords = coords[corners]
        area = 0.5 * np.abs(np.dot(corner_coords[:, 0], np.roll(corner_coords[:, 1], 1)) - np.dot(corner_coords[:, 1], np.roll(corner_coords[:, 0], 1)))
        assert area > 0
        midsides = elem[4:]
        expected_midsides = [0.5 * (corner_coords[0] + corner_coords[1]), 0.5 * (corner_coords[1] + corner_coords[2]), 0.5 * (corner_coords[2] + corner_coords[3]), 0.5 * (corner_coords[3] + corner_coords[0])]
        for (i, mid) in enumerate(midsides):
            assert np.allclose(coords[mid], expected_midsides[i])
    for i in range(ny):
        for j in range(nx - 1):
            elem_right = connect[i * nx + j]
            elem_left = connect[i * nx + j + 1]
            assert elem_right[1] == elem_left[0]
            assert elem_right[2] == elem_left[3]
            assert elem_right[6] == elem_left[7]
    for i in range(ny - 1):
        for j in range(nx):
            elem_top = connect[i * nx + j]
            elem_bottom = connect[(i + 1) * nx + j]
            assert elem_top[3] == elem_bottom[0]
            assert elem_top[2] == elem_bottom[1]
            assert elem_top[7] == elem_bottom[4]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
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