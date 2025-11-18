def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_nodes = npx * npy
    expected_elements = 2 * nx * ny
    assert coords.shape[0] == expected_nodes
    assert connect.shape[0] == expected_elements
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    id_bl = 0 * npx + 0
    id_br = 0 * npx + (npx - 1)
    id_tl = (npy - 1) * npx + 0
    id_tr = (npy - 1) * npx + (npx - 1)
    assert np.allclose(coords[id_bl], [xl, yl])
    assert np.allclose(coords[id_br], [xh, yl])
    assert np.allclose(coords[id_tl], [xl, yh])
    assert np.allclose(coords[id_tr], [xh, yh])
    x_expected = np.linspace(xl, xh, npx)
    y_expected = np.linspace(yl, yh, npy)
    (xx, yy) = np.meshgrid(x_expected, y_expected, indexing='xy')
    coords_expected = np.vstack([xx.ravel(), yy.ravel()]).T
    assert np.allclose(coords, coords_expected)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 0.0, 3.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < n_nodes)
    for row in connect:
        assert len(set(row)) == 6
    for elem in connect:
        (n1, n2, n3, n4, n5, n6) = elem
        (c1, c2, c3) = (coords[n1], coords[n2], coords[n3])
        (c4, c5, c6) = (coords[n4], coords[n5], coords[n6])
        area = 0.5 * (c1[0] * (c2[1] - c3[1]) + c2[0] * (c3[1] - c1[1]) + c3[0] * (c1[1] - c2[1]))
        assert area > 1e-09
        assert np.allclose(c4, 0.5 * (c1 + c2))
        assert np.allclose(c5, 0.5 * (c2 + c3))
        assert np.allclose(c6, 0.5 * (c3 + c1))
    for i in range(0, connect.shape[0], 2):
        elem1 = connect[i]
        elem2 = connect[i + 1]
        assert elem1[1] == elem2[1]
        assert elem1[0] == elem2[2]
        assert elem1[3] == elem2[4]

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (1, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, ny)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, -1, ny)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, nx, 0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, nx, -1)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(1.1, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.1, xh, 1.0, nx, ny)