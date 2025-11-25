def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_elements = nx * ny
    assert coords.shape[0] == expected_nodes
    assert connect.shape[0] == expected_elements
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 8)
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
    assert len(unique_x) == 2 * nx + 1
    assert len(unique_y) == 2 * ny + 1
    assert np.allclose(np.sort(unique_x), expected_x)
    assert np.allclose(np.sort(unique_y), expected_y)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 4.0, 5.0)
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
        area = 0.5 * abs(sum((x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4))))
        assert area > 0
        v1 = corners[1] - corners[0]
        v2 = corners[3] - corners[0]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross > 0
    for elem in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem
        expected_n5 = (coords[n1] + coords[n2]) / 2
        assert np.allclose(coords[n5], expected_n5)
        expected_n6 = (coords[n2] + coords[n3]) / 2
        assert np.allclose(coords[n6], expected_n6)
        expected_n7 = (coords[n3] + coords[n4]) / 2
        assert np.allclose(coords[n7], expected_n7)
        expected_n8 = (coords[n4] + coords[n1]) / 2
        assert np.allclose(coords[n8], expected_n8)
    for cy in range(ny):
        for cx in range(nx - 1):
            left_elem = cy * nx + cx
            right_elem = cy * nx + (cx + 1)
            left_n2 = connect[left_elem, 1]
            left_n3 = connect[left_elem, 2]
            left_n6 = connect[left_elem, 5]
            right_n1 = connect[right_elem, 0]
            right_n4 = connect[right_elem, 3]
            right_n8 = connect[right_elem, 7]
            assert left_n2 == right_n1
            assert left_n3 == right_n4
            assert left_n6 == right_n8
    for cy in range(ny - 1):
        for cx in range(nx):
            bottom_elem = cy * nx + cx
            top_elem = (cy + 1) * nx + cx
            bottom_n3 = connect[bottom_elem, 2]
            bottom_n4 = connect[bottom_elem, 3]
            bottom_n7 = connect[bottom_elem, 6]
            top_n1 = connect[top_elem, 0]
            top_n2 = connect[top_elem, 1]
            top_n5 = connect[top_elem, 4]
            assert bottom_n4 == top_n1
            assert bottom_n3 == top_n2
            assert bottom_n7 == top_n5

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, ny)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, -1, ny)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, nx, 0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, nx, -1)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(2.0, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 2.0, xh, 1.0, nx, ny)