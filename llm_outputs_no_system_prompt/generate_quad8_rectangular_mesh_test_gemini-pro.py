def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == (12, 2)
    assert connect.shape == (4, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.allclose(coords[0], [0.0, 0.0])
    assert np.allclose(coords[4], [2.0, 2.0])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    assert np.allclose(coords[1] - coords[0], [0.5 * dx, 0.0])
    assert np.allclose(coords[3] - coords[0], [0.0, 0.5 * dy])
    assert np.all(coords == coords2)
    assert np.all(connect == connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    for element in connect:
        assert np.all((element >= 0) & (element < n_nodes))
        assert len(np.unique(element)) == 8
    for element in connect:
        (n1, n2, n3, n4) = element[:4]
        area = 0.5 * ((coords[n2, 0] - coords[n1, 0]) * (coords[n3, 1] - coords[n1, 1]))
        assert area > 0
        (n5, n6, n7, n8) = element[4:]
        assert np.allclose(coords[n5], 0.5 * (coords[n1] + coords[n2]))
        assert np.allclose(coords[n6], 0.5 * (coords[n2] + coords[n3]))
        assert np.allclose(coords[n7], 0.5 * (coords[n3] + coords[n4]))
        assert np.allclose(coords[n8], 0.5 * (coords[n4] + coords[n1]))
    assert np.all(connect[0, [1, 5, 2]] == connect[1, [0, 7, 3]])
    assert np.all(connect[0, [3, 7, 0]] == connect[2, [1, 5, 2]])

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 1, 0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xl, yh, 1, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yl, 1, 1)