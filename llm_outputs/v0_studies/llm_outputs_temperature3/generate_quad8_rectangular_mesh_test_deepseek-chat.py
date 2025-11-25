def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    np.testing.assert_array_equal(coords1, coords2)
    np.testing.assert_array_equal(connect1, connect2)
    assert coords1.shape == (21, 2)
    assert connect1.shape == (4, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert len(coords1) == 21
    assert np.min(coords1[:, 0]) == 0.0
    assert np.max(coords1[:, 0]) == 1.0
    assert np.min(coords1[:, 1]) == 0.0
    assert np.max(coords1[:, 1]) == 1.0
    dx_half = 0.25
    dy_half = 0.25
    x_coords = np.unique(coords1[:, 0])
    y_coords = np.unique(coords1[:, 1])
    assert np.allclose(np.diff(x_coords), dx_half)
    assert np.allclose(np.diff(y_coords), dy_half)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 2.0, 1.0, 2, 1)
    assert np.min(connect) >= 0
    assert np.max(connect) < len(coords)
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        corners = elem[:4]
        midsides = elem[4:]
        corner_coords = coords[corners]
        area = 0.5 * np.abs(np.dot(corner_coords[:, 0], np.roll(corner_coords[:, 1], 1)) - np.dot(corner_coords[:, 1], np.roll(corner_coords[:, 0], 1)))
        assert area > 0
        expected_midsides = [0.5 * (corner_coords[0] + corner_coords[1]), 0.5 * (corner_coords[1] + corner_coords[2]), 0.5 * (corner_coords[2] + corner_coords[3]), 0.5 * (corner_coords[3] + corner_coords[0])]
        for (i, mid_idx) in enumerate(midsides):
            np.testing.assert_allclose(coords[mid_idx], expected_midsides[i])
    if connect.shape[0] > 1:
        elem1_right_edge = [connect[0, 1], connect[0, 2], connect[0, 6]]
        elem2_left_edge = [connect[1, 0], connect[1, 3], connect[1, 7]]
        assert set(elem1_right_edge) == set(elem2_left_edge)

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 2, 2)