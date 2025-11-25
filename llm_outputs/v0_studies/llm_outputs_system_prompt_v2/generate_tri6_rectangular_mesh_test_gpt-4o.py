def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == ((2 * nx + 1) * (2 * ny + 1), 2)
    assert connect.shape == (2 * nx * ny, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    (dx, dy) = ((xh - xl) / nx, (yh - yl) / ny)
    expected_coords = np.array([[xl, yl], [xl + dx / 2, yl], [xl + dx, yl], [xl, yl + dy / 2], [xl + dx / 2, yl + dy / 2], [xl + dx, yl + dy / 2], [xl, yl + dy], [xl + dx / 2, yl + dy], [xl + dx, yl + dy]])
    np.testing.assert_allclose(coords, expected_coords)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    np.testing.assert_array_equal(coords, coords1)
    np.testing.assert_array_equal(connect, connect1)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    assert np.all(connect >= 0)
    assert np.all(connect < npx * npy)
    for elem in connect:
        (N1, N2, N3, N4, N5, N6) = elem
        assert np.cross(coords[N2] - coords[N1], coords[N3] - coords[N1]) > 0
        np.testing.assert_allclose(coords[N4], (coords[N1] + coords[N2]) / 2)
        np.testing.assert_allclose(coords[N5], (coords[N2] + coords[N3]) / 2)
        np.testing.assert_allclose(coords[N6], (coords[N3] + coords[N1]) / 2)
    unique_edges = set()
    for elem in connect:
        edges = {(elem[i], elem[j]) for (i, j) in [(0, 1), (1, 2), (2, 0)]}
        for edge in edges:
            assert edge not in unique_edges
            unique_edges.add(edge)

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
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