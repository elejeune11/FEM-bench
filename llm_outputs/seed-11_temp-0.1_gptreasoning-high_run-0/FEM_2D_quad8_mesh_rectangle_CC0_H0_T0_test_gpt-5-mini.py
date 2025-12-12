def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert isinstance(coords1, np.ndarray)
    assert isinstance(connect1, np.ndarray)
    Nnodes_exp = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    Ne_exp = nx * ny
    assert coords1.shape == (Nnodes_exp, 2)
    assert connect1.shape == (Ne_exp, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for c in corners:
        found = np.any(np.all(np.isclose(coords1, c, rtol=0, atol=1e-12), axis=1))
        assert found
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    stepx = 0.5 * dx
    stepy = 0.5 * dy
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected = []
    for iy in range(npy):
        for ix in range(npx):
            if not (ix % 2 == 1 and iy % 2 == 1):
                expected.append([xl + stepx * ix, yl + stepy * iy])
    expected = np.array(expected, dtype=np.float64)
    assert expected.shape == coords1.shape
    assert np.allclose(coords1, expected, rtol=1e-12, atol=1e-12)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 0.5, 2.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    Ne = nx * ny
    assert coords.shape == (Nnodes, 2)
    assert connect.shape == (Ne, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    for e in range(Ne):
        row = connect[e]
        assert len(np.unique(row)) == 8
        p1 = coords[row[0]]
        p2 = coords[row[1]]
        p3 = coords[row[2]]
        p4 = coords[row[3]]
        (x0, y0) = p1
        (x1, y1) = p2
        (x2, y2) = p3
        (x3, y3) = p4
        area = 0.5 * (x0 * y1 - x1 * y0 + (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3))
        assert area > 0
        assert np.allclose(coords[row[4]], 0.5 * (p1 + p2), rtol=1e-12, atol=1e-12)
        assert np.allclose(coords[row[5]], 0.5 * (p2 + p3), rtol=1e-12, atol=1e-12)
        assert np.allclose(coords[row[6]], 0.5 * (p3 + p4), rtol=1e-12, atol=1e-12)
        assert np.allclose(coords[row[7]], 0.5 * (p4 + p1), rtol=1e-12, atol=1e-12)
    for cy in range(ny):
        for cx in range(nx):
            e = cy * nx + cx
            if cx < nx - 1:
                left = connect[e]
                right = connect[e + 1]
                assert left[1] == right[0]
                assert left[5] == right[7]
                assert left[2] == right[3]
            if cy < ny - 1:
                bottom = connect[e]
                top = connect[e + nx]
                assert bottom[2] == top[0]
                assert bottom[6] == top[4]
                assert bottom[3] == top[1]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)