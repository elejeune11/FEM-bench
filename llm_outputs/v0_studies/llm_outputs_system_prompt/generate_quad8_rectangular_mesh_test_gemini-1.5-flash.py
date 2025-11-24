def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == ((2 * nx + 1) * (2 * ny + 1) - nx * ny, 2)
    assert coords.dtype == np.float64
    assert connect.shape == (nx * ny, 8)
    assert connect.dtype == np.int64
    assert np.all(coords[connect[:, 0], 0] == xl)
    assert np.all(coords[connect[:, 1], 0] == xh)
    assert np.all(coords[connect[:, 2], 0] == xh)
    assert np.all(coords[connect[:, 3], 0] == xl)
    assert np.all(coords[connect[:, 0], 1] == yl)
    assert np.all(coords[connect[:, 1], 1] == yl)
    assert np.all(coords[connect[:, 2], 1] == yh)
    assert np.all(coords[connect[:, 3], 1] == yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.allclose(coords, coords2)
    assert np.allclose(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all((connect >= 0) & (connect < coords.shape[0]))
    assert np.all(np.diff(np.sort(connect.flatten())) > 0)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    for i in range(nx * ny):
        (n1, n2, n3, n4, n5, n6, n7, n8) = connect[i]
        assert np.allclose(coords[n5], 0.5 * (coords[n1] + coords[n2]))
        assert np.allclose(coords[n6], 0.5 * (coords[n2] + coords[n3]))
        assert np.allclose(coords[n7], 0.5 * (coords[n3] + coords[n4]))
        assert np.allclose(coords[n8], 0.5 * (coords[n4] + coords[n1]))

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    try:
        fcn(xl, yl, xh, yh, 0, 2)
        assert False
    except ValueError:
        pass
    try:
        fcn(xl, yl, xh, yh, 2, 0)
        assert False
    except ValueError:
        pass
    try:
        fcn(xl, yl, xh, yh, 2, 2)
        assert False
    except ValueError:
        pass
    try:
        fcn(2.0, 0.0, 0.0, 2.0, 2, 2)
        assert False
    except ValueError:
        pass