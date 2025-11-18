def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    ne_expected = nx * ny
    nnodes_expected = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    assert coords.shape[0] == nnodes_expected
    assert connect.shape[0] == ne_expected
    assert coords.shape == (nnodes_expected, 2)
    assert connect.shape == (ne_expected, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    domain_corners = np.array([[xl, yl], [xh, yl], [xl, yh], [xh, yh]])
    for corner in domain_corners:
        assert np.any(np.all(np.isclose(coords, corner), axis=1))
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    x_ticks = np.linspace(xl, xh, npx)
    y_ticks = np.linspace(yl, yh, npy)
    (xv, yv) = np.meshgrid(x_ticks, y_ticks, indexing='xy')
    all_coords = np.vstack((xv.ravel(), yv.ravel())).T
    mask = []
    for iy in range(npy):
        for ix in range(npx):
            is_center = ix % 2 == 1 and iy % 2 == 1
            mask.append(not is_center)
    expected_coords = all_coords[mask]
    assert np.allclose(coords, expected_coords)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 0.0, 3.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < nnodes)
    for row in connect:
        assert len(np.unique(row)) == 8, 'Node IDs within an element must be unique'
    corner_indices = connect[:, :4]
    for el_corners in corner_indices:
        p = coords[el_corners]
        area = 0.5 * (p[0, 0] * p[1, 1] - p[1, 0] * p[0, 1] + p[1, 0] * p[2, 1] - p[2, 0] * p[1, 1] + p[2, 0] * p[3, 1] - p[3, 0] * p[2, 1] + p[3, 0] * p[0, 1] - p[0, 0] * p[3, 1])
        assert area > 1e-09, 'Corner nodes are not ordered counter-clockwise'
    for el_nodes in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = el_nodes
        (c1, c2, c3, c4) = coords[[n1, n2, n3, n4]]
        (c5, c6, c7, c8) = coords[[n5, n6, n7, n8]]
        assert np.allclose(c5, 0.5 * (c1 + c2))
        assert np.allclose(c6, 0.5 * (c2 + c3))
        assert np.allclose(c7, 0.5 * (c3 + c4))
        assert np.allclose(c8, 0.5 * (c4 + c1))
    el1_idx = 0 * nx + 0
    el2_idx = 0 * nx + 1
    el1 = connect[el1_idx]
    el2 = connect[el2_idx]
    assert el1[1] == el2[0]
    assert el1[2] == el2[3]
    assert el1[5] == el2[7]
    el3_idx = 1 * nx + 0
    el3 = connect[el3_idx]
    assert el1[3] == el3[0]
    assert el1[2] == el3[1]
    assert el1[6] == el3[4]

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