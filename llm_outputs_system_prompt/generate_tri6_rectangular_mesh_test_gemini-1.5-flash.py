def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords1.shape == (25, 2)
    assert connect1.shape == (8, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.allclose(coords1, coords2)
    assert np.allclose(connect1, connect2)
    assert np.allclose(coords1[0, :], [0.0, 0.0])
    assert np.allclose(coords1[4, :], [1.0, 0.0])
    assert np.allclose(coords1[20, :], [0.0, 2.0])
    assert np.allclose(coords1[24, :], [2.0, 2.0])
    assert np.allclose(coords1[1, :], [0.5, 0.0])
    assert np.allclose(coords1[5, :], [1.5, 0.0])
    assert np.allclose(coords1[21, :], [0.5, 2.0])
    assert np.allclose(coords1[25 - 1, :], [1.5, 2.0])

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    assert np.all((connect >= 0) & (connect < npx * npy))
    assert np.all(np.diff(np.sort(connect.flatten())) > 0)

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
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
    try:
        fcn(0.0, 2.0, 2.0, 0.0, 2, 2)
        assert False
    except ValueError:
        pass