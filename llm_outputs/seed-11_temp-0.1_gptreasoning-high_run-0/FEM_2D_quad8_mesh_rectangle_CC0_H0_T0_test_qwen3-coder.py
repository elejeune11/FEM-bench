def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    np.testing.assert_array_equal(coords1, coords2)
    np.testing.assert_array_equal(connect1, connect2)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_elements = nx * ny
    assert coords1.shape == (expected_nodes, 2)
    assert connect1.shape == (expected_elements, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.any(coords1[:, 0] == xl)
    assert np.any(coords1[:, 0] == xh)
    assert np.any(coords1[:, 1] == yl)
    assert np.any(coords1[:, 1] == yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_positions = np.unique(coords1[:, 0])
    y_positions = np.unique(coords1[:, 1])
    expected_x = np.linspace(xl, xh, 2 * nx + 1)
    expected_y = np.linspace(yl, yh, 2 * ny + 1)
    np.testing.assert_allclose(x_positions, expected_x)
    np.testing.assert_allclose(y_positions, expected_y)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0)
    assert np.all(connect < len(coords))
    for row in connect:
        assert len(np.unique(row)) == len(row)
    for elem in connect:
        c1 = coords[elem[0]]
        c2 = coords[elem[1]]
        c3 = coords[elem[2]]
        c4 = coords[elem[3]]
        area = 0.5 * (c1[0] * c2[1] - c2[0] * c1[1] + (c2[0] * c3[1] - c3[0] * c2[1]) + (c3[0] * c4[1] - c4[0] * c3[1]) + (c4[0] * c1[1] - c1[0] * c4[1]))
        assert area > 0
    for elem in connect:
        (c1, c2, c3, c4) = (coords[elem[0]], coords[elem[1]], coords[elem[2]], coords[elem[3]])
        (m5, m6, m7, m8) = (coords[elem[4]], coords[elem[5]], coords[elem[6]], coords[elem[7]])
        np.testing.assert_allclose(m5, 0.5 * (c1 + c2))
        np.testing.assert_allclose(m6, 0.5 * (c2 + c3))
        np.testing.assert_allclose(m7, 0.5 * (c3 + c4))
        np.testing.assert_allclose(m8, 0.5 * (c4 + c1))
    elem0 = connect[0]
    elem1 = connect[1]
    assert elem0[1] == elem1[0]
    assert elem0[5] == elem1[4]
    assert elem0[2] == elem1[3]

def test_quad8_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 1, 0)
    with pytest.raises(ValueError):
        fcn(xh, yl, xl, yh, 1, 1)
    with pytest.raises(ValueError):
        fcn(xl, yh, xh, yl, 1, 1)