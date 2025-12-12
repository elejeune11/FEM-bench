def test_quad8_mesh_basic_structure_and_determinism(fcn):
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == (12, 2)
    assert connect.shape == (4, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert coords.shape == (21, 2)
    assert connect.shape == (4, 8)
    assert np.isclose(coords[0, 0], xl) and np.isclose(coords[0, 1], yl)
    assert np.isclose(coords[4, 0], xh) and np.isclose(coords[4, 1], yl)
    assert np.isclose(coords[20, 0], xh) and np.isclose(coords[20, 1], yh)
    assert np.isclose(coords[16, 0], xl) and np.isclose(coords[16, 1], yh)
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    expected_x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    expected_y = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.allclose(unique_x, expected_x)
    assert np.allclose(unique_y, expected_y)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    (xl, yl, xh, yh) = (0.0, 0.0, 4.0, 2.0)
    (nx, ny) = (2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0) and np.all(connect < coords.shape[0])
    for elem in connect:
        (n1, n2, n3, n4) = (coords[elem[0]], coords[elem[1]], coords[elem[2]], coords[elem[3]])
        area = n1[0] * n2[1] - n2[0] * n1[1] + (n2[0] * n3[1] - n3[0] * n2[1]) + (n3[0] * n4[1] - n4[0] * n3[1]) + (n4[0] * n1[1] - n1[0] * n4[1])
        assert area > 0
    for elem in connect:
        (n1, n2, n3, n4) = (coords[elem[0]], coords[elem[1]], coords[elem[2]], coords[elem[3]])
        (n5, n6, n7, n8) = (coords[elem[4]], coords[elem[5]], coords[elem[6]], coords[elem[7]])
        assert np.allclose(n5, (n1 + n2) / 2.0)
        assert np.allclose(n6, (n2 + n3) / 2.0)
        assert np.allclose(n7, (n3 + n4) / 2.0)
        assert np.allclose(n8, (n4 + n1) / 2.0)
    shared_node_index = 6
    assert shared_node_index in connect[0] and shared_node_index in connect[1]
    assert np.isclose(coords[shared_node_index, 0], 2.0) and np.isclose(coords[shared_node_index, 1], 1.0)

def test_quad8_mesh_invalid_inputs(fcn):
    with pytest.raises(ValueError):
        fcn(0, 0, 1, 1, 0, 1)
    with pytest.raises(ValueError):
        fcn(0, 0, 1, 1, -1, 1)
    with pytest.raises(ValueError):
        fcn(0, 0, 1, 1, 1, 0)
    with pytest.raises(ValueError):
        fcn(0, 0, 1, 1, 1, -1)
    with pytest.raises(ValueError):
        fcn(1, 0, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        fcn(2, 0, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        fcn(0, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        fcn(0, 2, 1, 1, 1, 1)