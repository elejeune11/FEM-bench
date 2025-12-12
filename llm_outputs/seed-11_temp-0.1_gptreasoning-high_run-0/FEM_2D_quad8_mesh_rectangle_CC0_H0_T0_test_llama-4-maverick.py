def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    Nnodes = npx * npy - nx * ny
    Ne = nx * ny
    assert coords1.shape == (Nnodes, 2)
    assert connect1.shape == (Ne, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.allclose(coords1, coords2)
    assert np.allclose(connect1, connect2)
    (dx, dy) = ((xh - xl) / nx, (yh - yl) / ny)
    assert np.isclose(coords1[:, 0].min(), xl)
    assert np.isclose(coords1[:, 0].max(), xh)
    assert np.isclose(coords1[:, 1].min(), yl)
    assert np.isclose(coords1[:, 1].max(), yh)
    assert np.allclose(np.diff(np.unique(coords1[:, 0])), 0.5 * dx)
    assert np.allclose(np.diff(np.unique(coords1[:, 1])), 0.5 * dy)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 0.5, 2.0, 3.5)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < Nnodes)
    for row in connect:
        assert len(np.unique(row)) == 8
        (N1, N2, N3, N4) = coords[row[:4]]
        area = 0.5 * np.abs(np.cross(N2 - N1, N3 - N1) + np.cross(N4 - N1, N3 - N1))
        assert area > 0
        (N5, N6, N7, N8) = coords[row[4:]]
        assert np.allclose(N5, 0.5 * (N1 + N2))
        assert np.allclose(N6, 0.5 * (N2 + N3))
        assert np.allclose(N7, 0.5 * (N3 + N4))
        assert np.allclose(N8, 0.5 * (N4 + N1))
    for cx in range(nx - 1):
        left_elem = connect[cx]
        right_elem = connect[cx + 1]
        assert left_elem[2] == right_elem[0]
        assert left_elem[3] == right_elem[7]
        assert left_elem[6] == right_elem[4]
    for cy in range(ny - 1):
        bottom_elem = connect[cy * nx]
        top_elem = connect[(cy + 1) * nx]
        assert bottom_elem[4] == top_elem[8]
        assert bottom_elem[1] == top_elem[5]
        assert bottom_elem[2] == top_elem[6]

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