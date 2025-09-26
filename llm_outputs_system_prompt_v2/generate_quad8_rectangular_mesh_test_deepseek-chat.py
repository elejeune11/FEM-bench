def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements."""
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords1.shape == (21, 2)
    assert connect1.shape == (4, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    assert np.allclose(coords1[0], [0.0, 0.0])
    assert np.allclose(coords1[4], [1.0, 0.0])
    assert np.allclose(coords1[20], [1.0, 1.0])
    assert np.allclose(coords1[16], [0.0, 1.0])
    dx_actual = coords1[1, 0] - coords1[0, 0]
    dy_actual = coords1[5, 1] - coords1[0, 1]
    assert np.isclose(dx_actual, 0.25)
    assert np.isclose(dy_actual, 0.25)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements."""
    (coords, connect) = fcn(1.0, 2.0, 3.0, 5.0, 2, 3)
    assert np.all(connect >= 0)
    assert np.all(connect < len(coords))
    for elem in connect:
        corners = elem[:4]
        assert len(np.unique(corners)) == 4
        x_corners = coords[corners, 0]
        y_corners = coords[corners, 1]
        area = 0.5 * np.abs(np.dot(x_corners, np.roll(y_corners, 1)) - np.dot(y_corners, np.roll(x_corners, 1)))
        assert area > 0
        n5 = coords[elem[4]]
        n1_n2_avg = (coords[elem[0]] + coords[elem[1]]) / 2
        assert np.allclose(n5, n1_n2_avg)
        n6 = coords[elem[5]]
        n2_n3_avg = (coords[elem[1]] + coords[elem[2]]) / 2
        assert np.allclose(n6, n2_n3_avg)
        n7 = coords[elem[6]]
        n3_n4_avg = (coords[elem[2]] + coords[elem[3]]) / 2
        assert np.allclose(n7, n3_n4_avg)
        n8 = coords[elem[7]]
        n4_n1_avg = (coords[elem[3]] + coords[elem[0]]) / 2
        assert np.allclose(n8, n4_n1_avg)
    for i in range(connect.shape[0] - 1):
        for j in range(i + 1, connect.shape[0]):
            shared_nodes = np.intersect1d(connect[i], connect[j])
            assert len(shared_nodes) > 0

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation."""
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)