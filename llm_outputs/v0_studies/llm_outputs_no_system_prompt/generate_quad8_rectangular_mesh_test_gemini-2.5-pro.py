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
    assert np.isclose(np.min(coords[:, 0]), xl)
    assert np.isclose(np.max(coords[:, 0]), xh)
    assert np.isclose(np.min(coords[:, 1]), yl)
    assert np.isclose(np.max(coords[:, 1]), yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    x_pts = xl + 0.5 * dx * np.arange(npx)
    y_pts = yl + 0.5 * dy * np.arange(npy)
    expected_coords_list = []
    for iy in range(npy):
        for ix in range(npx):
            if not (ix % 2 == 1 and iy % 2 == 1):
                expected_coords_list.append([x_pts[ix], y_pts[iy]])
    expected_coords = np.array(expected_coords_list, dtype=np.float64)
    assert np.allclose(coords, expected_coords), 'Node coordinates do not match expected lattice'
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
    nnodes = coords.shape[0]
    assert np.all(connect >= 0) and np.all(connect < nnodes)
    for row in connect:
        assert len(np.unique(row)) == 8, 'Node indices within an element are not unique'
    corner_indices = connect[:, [0, 1, 2, 3]]
    corner_coords = coords[corner_indices]
    x = corner_coords[:, :, 0]
    y = corner_coords[:, :, 1]
    area = 0.5 * (x[:, 0] * y[:, 1] - y[:, 0] * x[:, 1] + x[:, 1] * y[:, 2] - y[:, 1] * x[:, 2] + x[:, 2] * y[:, 3] - y[:, 2] * x[:, 3] + x[:, 3] * y[:, 0] - y[:, 3] * x[:, 0])
    assert np.all(area > 1e-09), 'Corner nodes are not consistently counter-clockwise'
    (n1, n2, n3, n4) = (connect[:, 0], connect[:, 1], connect[:, 2], connect[:, 3])
    (n5, n6, n7, n8) = (connect[:, 4], connect[:, 5], connect[:, 6], connect[:, 7])
    assert np.allclose(coords[n5], 0.5 * (coords[n1] + coords[n2]))
    assert np.allclose(coords[n6], 0.5 * (coords[n2] + coords[n3]))
    assert np.allclose(coords[n7], 0.5 * (coords[n3] + coords[n4]))
    assert np.allclose(coords[n8], 0.5 * (coords[n4] + coords[n1]))
    for cy in range(ny):
        for cx in range(nx - 1):
            e1_idx = cy * nx + cx
            e2_idx = cy * nx + (cx + 1)
            assert connect[e1_idx, 1] == connect[e2_idx, 0]
            assert connect[e1_idx, 2] == connect[e2_idx, 3]
            assert connect[e1_idx, 5] == connect[e2_idx, 7]
    for cy in range(ny - 1):
        for cx in range(nx):
            e1_idx = cy * nx + cx
            e2_idx = (cy + 1) * nx + cx
            assert connect[e1_idx, 3] == connect[e2_idx, 0]
            assert connect[e1_idx, 2] == connect[e2_idx, 1]
            assert connect[e1_idx, 6] == connect[e2_idx, 4]

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
        fcn(1.1, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.1, 1.0, 1.0, 1, 1)