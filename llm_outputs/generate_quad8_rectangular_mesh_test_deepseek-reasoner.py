def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords1.shape == (21, 2)
    assert connect1.shape == (4, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert len(coords1) == 21
    assert len(connect1) == 4
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for corner in corners:
        assert any(np.allclose(coords1, corner, atol=1e-10))
    (dx, dy) = (0.5 / 2, 0.5 / 2)
    for coord in coords1:
        x_rem = coord[0] % dx
        y_rem = coord[1] % dy
        assert abs(x_rem) < 1e-10 or abs(x_rem - dx) < 1e-10
        assert abs(y_rem) < 1e-10 or abs(y_rem - dy) < 1e-10
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (coords, connect) = fcn(1.0, 2.0, 5.0, 6.0, 3, 2)
    assert connect.min() >= 0
    assert connect.max() < len(coords)
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        corners = coords[elem[:4]]
        midsides = coords[elem[4:]]
        (x, y) = (corners[:, 0], corners[:, 1])
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        assert area > 0
        expected_midsides = [0.5 * (corners[0] + corners[1]), 0.5 * (corners[1] + corners[2]), 0.5 * (corners[2] + corners[3]), 0.5 * (corners[3] + corners[0])]
        for (actual, expected) in zip(midsides, expected_midsides):
            assert np.allclose(actual, expected, atol=1e-10)
    for i in range(len(connect) - 1):
        elem1 = connect[i]
        if (i + 1) % 3 != 0:
            elem2 = connect[i + 1]
            assert elem1[1] == elem2[3]
            assert elem1[5] == elem2[7]
            assert elem1[2] == elem2[0]
        if i < 3:
            elem2 = connect[i + 3]
            assert elem1[2] == elem2[0]
            assert elem1[6] == elem2[4]
            assert elem1[3] == elem2[1]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 2, 2)