def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == ((2 * nx + 1) * (2 * ny + 1), 2), 'Incorrect coordinates shape'
    assert connect.shape == (2 * nx * ny, 6), 'Incorrect connectivity shape'
    assert coords.dtype == np.float64, 'Incorrect coordinates dtype'
    assert connect.dtype == np.int64, 'Incorrect connectivity dtype'
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    for ix in range(npx):
        for iy in range(npy):
            node_id = iy * npx + ix
            x_expected = xl + 0.5 * (xh - xl) * ix / (npx - 1)
            y_expected = yl + 0.5 * (yh - yl) * iy / (npy - 1)
            assert np.allclose(coords[node_id], [x_expected, y_expected]), 'Incorrect node coordinates'
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2), 'Non-deterministic coordinates'
    assert np.array_equal(connect, connect2), 'Non-deterministic connectivity'

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    for e in range(connect.shape[0]):
        elem = connect[e]
        for node_id in elem:
            assert 0 <= node_id < npx * npy, 'Invalid node ID'
    for e in range(connect.shape[0]):
        (N1, N2, N3, N4, N5, N6) = connect[e]
        (p1, p2, p3) = (coords[N1], coords[N2], coords[N3])
        assert (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) > 0, 'Non-counter-clockwise ordering'
        assert np.allclose(coords[N4], (coords[N1] + coords[N2]) / 2), 'Incorrect midside node N4'
        assert np.allclose(coords[N5], (coords[N2] + coords[N3]) / 2), 'Incorrect midside node N5'
        assert np.allclose(coords[N6], (coords[N3] + coords[N1]) / 2), 'Incorrect midside node N6'

def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)