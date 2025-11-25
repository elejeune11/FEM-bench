def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_ne = nx * ny
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    assert coords.shape[0] == expected_nnodes, 'Incorrect number of nodes'
    assert connect.shape[0] == expected_ne, 'Incorrect number of elements'
    assert coords.shape == (expected_nnodes, 2), 'Incorrect coords shape'
    assert connect.shape == (expected_ne, 8), 'Incorrect connect shape'
    assert coords.dtype == np.float64, 'Incorrect coords dtype'
    assert connect.dtype == np.int64, 'Incorrect connect dtype'
    assert np.isclose(coords[:, 0].min(), xl)
    assert np.isclose(coords[:, 0].max(), xh)
    assert np.isclose(coords[:, 1].min(), yl)
    assert np.isclose(coords[:, 1].max(), yh)
    domain_corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    for corner in domain_corners:
        assert np.any(np.all(np.isclose(coords, corner), axis=1)), 'Domain corner missing'
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    steps_x = (coords[:, 0] - xl) / half_dx
    steps_y = (coords[:, 1] - yl) / half_dy
    assert np.all(np.isclose(steps_x, np.round(steps_x))), 'X-coordinates not on half-step grid'
    assert np.all(np.isclose(steps_y, np.round(steps_y))), 'Y-coordinates not on half-step grid'
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2), 'Coords are not deterministic'
    assert np.array_equal(connect, connect2), 'Connectivity is not deterministic'

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 5.0, 8.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    assert np.all(connect >= 0) and np.all(connect < n_nodes)
    for i in range(connect.shape[0]):
        assert len(np.unique(connect[i, :])) == 8, 'Node IDs not unique within an element'
    for elem in connect:
        (n1, n2, n3, n4) = elem[0:4]
        (c1, c2, c3, c4) = coords[[n1, n2, n3, n4]]
        area = 0.5 * (c1[0] * c2[1] - c1[1] * c2[0] + (c2[0] * c3[1] - c2[1] * c3[0]) + (c3[0] * c4[1] - c3[1] * c4[0]) + (c4[0] * c1[1] - c4[1] * c1[0]))
        assert area > 1e-09, 'Element corners are not ordered counter-clockwise'
        (n5, n6, n7, n8) = elem[4:8]
        (c5, c6, c7, c8) = coords[[n5, n6, n7, n8]]
        assert np.allclose(c5, 0.5 * (c1 + c2)), 'Midside node N5 position is incorrect'
        assert np.allclose(c6, 0.5 * (c2 + c3)), 'Midside node N6 position is incorrect'
        assert np.allclose(c7, 0.5 * (c3 + c4)), 'Midside node N7 position is incorrect'
        assert np.allclose(c8, 0.5 * (c4 + c1)), 'Midside node N8 position is incorrect'
    for cy in range(ny):
        for cx in range(nx):
            if cx < nx - 1:
                elem_left = connect[cy * nx + cx]
                elem_right = connect[cy * nx + (cx + 1)]
                assert elem_left[1] == elem_right[0]
                assert elem_left[2] == elem_right[3]
                assert elem_left[5] == elem_right[7]
            if cy < ny - 1:
                elem_bottom = connect[cy * nx + cx]
                elem_top = connect[(cy + 1) * nx + cx]
                assert elem_bottom[3] == elem_top[0]
                assert elem_bottom[2] == elem_top[1]
                assert elem_bottom[6] == elem_top[4]

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
        fcn(xl, yl, xh, yh, nx, -5)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(2.0, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 2.0, xh, 1.0, nx, ny)