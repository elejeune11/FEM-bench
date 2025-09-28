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
    expected_x_vals = np.array([xl + 0.5 * dx * i for i in range(2 * nx + 1)])
    expected_y_vals = np.array([yl + 0.5 * dy * i for i in range(2 * ny + 1)])
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    assert np.allclose(np.sort(unique_x), expected_x_vals)
    assert np.allclose(np.sort(unique_y), expected_y_vals)
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
    max_node_id = coords.shape[0] - 1
    assert np.all(connect >= 0)
    assert np.all(connect <= max_node_id)
    for elem_nodes in connect:
        assert len(np.unique(elem_nodes)) == 8
    for elem_nodes in connect:
        corners = coords[elem_nodes[:4]]
        x = corners[:, 0]
        y = corners[:, 1]
        area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])
        assert area > 0
    for elem_nodes in connect:
        (n1, n2, n3, n4) = elem_nodes[:4]
        (n5, n6, n7, n8) = elem_nodes[4:]
        expected_n5 = 0.5 * (coords[n1] + coords[n2])
        assert np.allclose(coords[n5], expected_n5)
        expected_n6 = 0.5 * (coords[n2] + coords[n3])
        assert np.allclose(coords[n6], expected_n6)
        expected_n7 = 0.5 * (coords[n3] + coords[n4])
        assert np.allclose(coords[n7], expected_n7)
        expected_n8 = 0.5 * (coords[n4] + coords[n1])
        assert np.allclose(coords[n8], expected_n8)
    if nx > 1:
        for cy in range(ny):
            for cx in range(nx - 1):
                left_elem = cy * nx + cx
                right_elem = cy * nx + (cx + 1)
                left_right_edge = [connect[left_elem][1], connect[left_elem][2], connect[left_elem][5]]
                right_left_edge = [connect[right_elem][0], connect[right_elem][3], connect[right_elem][7]]
                assert left_right_edge[0] == right_left_edge[0]
                assert left_right_edge[1] == right_left_edge[1]
                assert left_right_edge[2] == right_left_edge[2]
    if ny > 1:
        for cy in range(ny - 1):
            for cx in range(nx):
                bottom_elem = cy * nx + cx
                top_elem = (cy + 1) * nx + cx
                bottom_top_edge = [connect[bottom_elem][3], connect[bottom_elem][2], connect[bottom_elem][6]]
                top_bottom_edge = [connect[top_elem][0], connect[top_elem][1], connect[top_elem][4]]
                assert bottom_top_edge[0] == top_bottom_edge[0]
                assert bottom_top_edge[1] == top_bottom_edge[1]
                assert bottom_top_edge[2] == top_bottom_edge[2]

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