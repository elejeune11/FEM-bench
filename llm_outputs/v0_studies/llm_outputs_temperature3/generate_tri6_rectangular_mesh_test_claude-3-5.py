def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords1.shape == (25, 2)
    assert connect1.shape == (8, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    np.testing.assert_array_equal(coords1, coords2)
    np.testing.assert_array_equal(connect1, connect2)
    assert np.allclose(np.min(coords1[:, 0]), 0.0)
    assert np.allclose(np.max(coords1[:, 0]), 1.0)
    assert np.allclose(np.min(coords1[:, 1]), 0.0)
    assert np.allclose(np.max(coords1[:, 1]), 1.0)
    x_coords = np.unique(coords1[:, 0])
    y_coords = np.unique(coords1[:, 1])
    assert len(x_coords) == 5
    assert len(y_coords) == 5
    assert np.allclose(np.diff(x_coords), 0.25)
    assert np.allclose(np.diff(y_coords), 0.25)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (coords, connect) = fcn(-1.0, 2.0, 2.0, 4.0, 2, 1)
    assert np.all(connect >= 0)
    assert np.all(connect < len(coords))
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        assert np.allclose(coords[elem[3]], 0.5 * (coords[elem[0]] + coords[elem[1]]))
        assert np.allclose(coords[elem[4]], 0.5 * (coords[elem[1]] + coords[elem[2]]))
        assert np.allclose(coords[elem[5]], 0.5 * (coords[elem[2]] + coords[elem[0]]))
    for elem in connect:
        v1 = coords[elem[1]] - coords[elem[0]]
        v2 = coords[elem[2]] - coords[elem[0]]
        cross = np.cross(v1, v2)
        assert cross > 0

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