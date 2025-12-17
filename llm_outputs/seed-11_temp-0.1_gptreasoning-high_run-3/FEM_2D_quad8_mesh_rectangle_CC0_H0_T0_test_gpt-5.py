def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure and determinism on a 2×2 unit square domain."""
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    nx, ny = (2, 2)
    coords1, connect1 = fcn(xl, yl, xh, yh, nx, ny)
    npx, npy = (2 * nx + 1, 2 * ny + 1)
    Nnodes = npx * npy - nx * ny
    Ne = nx * ny
    assert coords1.shape == (Nnodes, 2)
    assert connect1.shape == (Ne, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for c in corners:
        assert np.any(np.all(coords1 == c, axis=1))
    ux = np.unique(coords1[:, 0])
    uy = np.unique(coords1[:, 1])
    assert len(ux) == npx
    assert len(uy) == npy
    assert np.allclose(ux, np.linspace(xl, xh, npx))
    assert np.allclose(uy, np.linspace(yl, yh, npy))
    centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]], dtype=np.float64)
    for cc in centers:
        assert not np.any(np.all(coords1 == cc, axis=1))
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain with Quad8 elements."""
    xl, yl, xh, yh = (2.0, -1.0, 5.0, 3.0)
    nx, ny = (3, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    for e in range(connect.shape[0]):
        assert len(np.unique(connect[e])) == 8

    def polygon_area(p):
        x = p[:, 0]
        y = p[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    for e in range(connect.shape[0]):
        n = connect[e]
        corners = coords[n[[0, 1, 2, 3]]]
        area = polygon_area(corners)
        assert area > 0
        c1, c2, c3, c4 = (coords[n[0]], coords[n[1]], coords[n[2]], coords[n[3]])
        m5, m6, m7, m8 = (coords[n[4]], coords[n[5]], coords[n[6]], coords[n[7]])
        assert np.allclose(m5, 0.5 * (c1 + c2))
        assert np.allclose(m6, 0.5 * (c2 + c3))
        assert np.allclose(m7, 0.5 * (c3 + c4))
        assert np.allclose(m8, 0.5 * (c4 + c1))
    for cy in range(ny):
        for cx in range(nx - 1):
            eL = cy * nx + cx
            eR = cy * nx + (cx + 1)
            assert connect[eL, 1] == connect[eR, 0]
            assert connect[eL, 2] == connect[eR, 3]
    for cy in range(ny - 1):
        for cx in range(nx):
            eB = cy * nx + cx
            eT = (cy + 1) * nx + cx
            assert connect[eB, 3] == connect[eT, 0]
            assert connect[eB, 2] == connect[eT, 1]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate that invalid inputs raise ValueError."""
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
        fcn(0.0, 2.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 3.0, 1.0, 2.0, 1, 1)