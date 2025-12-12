def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    expected_n_nodes = npx * npy - nx * ny
    expected_n_elems = nx * ny
    assert coords.shape[0] == expected_n_nodes, 'Incorrect number of nodes'
    assert connect.shape[0] == expected_n_elems, 'Incorrect number of elements'
    assert coords.shape == (expected_n_nodes, 2), 'Incorrect coords shape'
    assert connect.shape == (expected_n_elems, 8), 'Incorrect connect shape'
    assert coords.dtype == np.float64, 'Incorrect coords dtype'
    assert connect.dtype == np.int64, 'Incorrect connect dtype'
    domain_corners = np.array([[xl, yl], [xh, yl], [xl, yh], [xh, yh]])
    for corner in domain_corners:
        assert np.any(np.all(np.isclose(coords, corner), axis=1)), 'Domain corner not found'
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    assert np.allclose(x_coords, np.round(x_coords / (0.5 * dx)) * (0.5 * dx))
    assert np.allclose(y_coords, np.round(y_coords / (0.5 * dy)) * (0.5 * dy))
    for cy in range(ny):
        for cx in range(nx):
            center_x = xl + (cx + 0.5) * dx
            center_y = yl + (cy + 0.5) * dy
            center_coord = np.array([center_x, center_y])
            assert not np.any(np.all(np.isclose(coords, center_coord), axis=1))
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2), 'Coordinates are not deterministic'
    assert np.array_equal(connect, connect2), 'Connectivity is not deterministic'

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 5.0, 8.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < n_nodes)
    for i in range(connect.shape[0]):
        assert len(np.unique(connect[i])) == 8, 'Node IDs in an element are not unique'
    for i in range(connect.shape[0]):
        elem_nodes = connect[i]
        elem_coords = coords[elem_nodes]
        (c1, c2, c3, c4) = elem_coords[0:4]
        (c5, c6, c7, c8) = elem_coords[4:8]
        area = 0.5 * (c1[0] * c2[1] - c1[1] * c2[0] + c2[0] * c3[1] - c2[1] * c3[0] + c3[0] * c4[1] - c3[1] * c4[0] + c4[0] * c1[1] - c4[1] * c1[0])
        assert area > 1e-09, 'Element corners are not in CCW order'
        assert np.allclose(c5, 0.5 * (c1 + c2)), 'Midside N5 is incorrect'
        assert np.allclose(c6, 0.5 * (c2 + c3)), 'Midside N6 is incorrect'
        assert np.allclose(c7, 0.5 * (c3 + c4)), 'Midside N7 is incorrect'
        assert np.allclose(c8, 0.5 * (c4 + c1)), 'Midside N8 is incorrect'
    for cy in range(ny):
        for cx in range(nx - 1):
            e1_idx = cy * nx + cx
            e2_idx = cy * nx + (cx + 1)
            e1 = connect[e1_idx]
            e2 = connect[e2_idx]
            assert e1[1] == e2[0]
            assert e1[2] == e2[3]
            assert e1[5] == e2[7]
    for cy in range(ny - 1):
        for cx in range(nx):
            e1_idx = cy * nx + cx
            e2_idx = (cy + 1) * nx + cx
            e1 = connect[e1_idx]
            e2 = connect[e2_idx]
            assert e1[3] == e2[0]
            assert e1[2] == e2[1]
            assert e1[6] == e2[4]

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