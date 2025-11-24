def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_nodes = npx * npy - nx * ny
    expected_elements = nx * ny
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert isinstance(coords, np.ndarray)
    assert isinstance(connect, np.ndarray)
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for c in corners:
        assert np.any(np.all(np.isclose(coords, c, rtol=0, atol=1e-14), axis=1))
    x_vals = np.unique(coords[:, 0])
    y_vals = np.unique(coords[:, 1])
    x_diffs = np.diff(np.sort(x_vals))
    y_diffs = np.diff(np.sort(y_vals))
    assert np.allclose(x_diffs, 0.5 * dx, rtol=0, atol=1e-14)
    assert np.allclose(y_diffs, 0.5 * dy, rtol=0, atol=1e-14)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-2.0, 1.5, 3.0, 5.5)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    for e in range(connect.shape[0]):
        row = connect[e]
        assert len(set(row.tolist())) == 8

    def poly_area_xy(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    for row in connect:
        corner_ids = row[0:4]
        pts = coords[corner_ids]
        area = poly_area_xy(pts)
        assert area > 0.0
    for row in connect:
        (N1, N2, N3, N4, N5, N6, N7, N8) = row
        assert np.allclose(coords[N5], 0.5 * (coords[N1] + coords[N2]), rtol=1e-12, atol=1e-12)
        assert np.allclose(coords[N6], 0.5 * (coords[N2] + coords[N3]), rtol=1e-12, atol=1e-12)
        assert np.allclose(coords[N7], 0.5 * (coords[N3] + coords[N4]), rtol=1e-12, atol=1e-12)
        assert np.allclose(coords[N8], 0.5 * (coords[N4] + coords[N1]), rtol=1e-12, atol=1e-12)

    def elem_index(cx, cy):
        return cy * nx + cx
    for cy in range(ny):
        for cx in range(nx - 1):
            eL = connect[elem_index(cx, cy)]
            eR = connect[elem_index(cx + 1, cy)]
            assert eL[0 + 1] == eR[0 + 0]
            assert eL[5] == eR[7]
            assert eL[2] == eR[3]
    for cy in range(ny - 1):
        for cx in range(nx):
            eB = connect[elem_index(cx, cy)]
            eT = connect[elem_index(cx, cy + 1)]
            assert eB[3] == eT[0]
            assert eB[6] == eT[4]
            assert eB[2] == eT[1]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -2, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -3)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)