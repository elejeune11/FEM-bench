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
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    domain_corners = [(xl, yl), (xh, yl), (xh, yh), (xl, yh)]
    for corner in domain_corners:
        distances = np.sqrt(np.sum((coords - corner) ** 2, axis=1))
        assert np.min(distances) < 1e-14
    expected_x = np.array([xl + 0.5 * dx * ix for ix in range(2 * nx + 1)])
    expected_y = np.array([yl + 0.5 * dy * iy for iy in range(2 * ny + 1)])
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    assert len(unique_x) == 2 * nx + 1
    assert len(unique_y) == 2 * ny + 1
    assert np.allclose(unique_x, expected_x)
    assert np.allclose(unique_y, expected_y)
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
    max_node_id = coords.shape[0] - 1
    assert np.all(connect >= 0)
    assert np.all(connect <= max_node_id)
    for elem_idx in range(connect.shape[0]):
        elem_nodes = connect[elem_idx]
        assert len(np.unique(elem_nodes)) == 8
    for elem_idx in range(connect.shape[0]):
        corners = connect[elem_idx, :4]
        corner_coords = coords[corners]
        x = corner_coords[:, 0]
        y = corner_coords[:, 1]
        area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1])
        assert area > 0
    for elem_idx in range(connect.shape[0]):
        elem_nodes = connect[elem_idx]
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
    for cy in range(ny):
        for cx in range(nx - 1):
            left_elem = cy * nx + cx
            right_elem = cy * nx + (cx + 1)
            left_right_edge = [connect[left_elem, 1], connect[left_elem, 2]]
            left_right_midside = connect[left_elem, 5]
            right_left_edge = [connect[right_elem, 0], connect[right_elem, 3]]
            right_left_midside = connect[right_elem, 7]
            assert left_right_edge == right_left_edge
            assert left_right_midside == right_left_midside
    for cy in range(ny - 1):
        for cx in range(nx):
            bottom_elem = cy * nx + cx
            top_elem = (cy + 1) * nx + cx
            bottom_top_edge = [connect[bottom_elem, 3], connect[bottom_elem, 2]]
            bottom_top_midside = connect[bottom_elem, 6]
            top_bottom_edge = [connect[top_elem, 0], connect[top_elem, 1]]
            top_bottom_midside = connect[top_elem, 4]
            assert bottom_top_edge == top_bottom_edge
            assert bottom_top_midside == top_bottom_midside

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