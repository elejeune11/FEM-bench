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
    assert np.min(coords1[:, 0]) == 0.0
    assert np.max(coords1[:, 0]) == 1.0
    assert np.min(coords1[:, 1]) == 0.0
    assert np.max(coords1[:, 1]) == 1.0
    x_coords = np.unique(coords1[:, 0])
    y_coords = np.unique(coords1[:, 1])
    expected_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected_y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(x_coords, expected_x)
    assert np.allclose(y_coords, expected_y)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 2.0, 3.0, 2, 3)
    n_nodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < n_nodes)
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        (n1, n2, n3, n4, n5, n6) = elem
        assert np.allclose(coords[n4], (coords[n1] + coords[n2]) / 2)
        assert np.allclose(coords[n5], (coords[n2] + coords[n3]) / 2)
        assert np.allclose(coords[n6], (coords[n3] + coords[n1]) / 2)
        v1 = coords[n2] - coords[n1]
        v2 = coords[n3] - coords[n1]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross_z > 0
    edge_dict = {}
    for (elem_idx, elem) in enumerate(connect):
        edges = [tuple(sorted([elem[0], elem[1]])), tuple(sorted([elem[1], elem[2]])), tuple(sorted([elem[2], elem[0]]))]
        for edge in edges:
            if edge in edge_dict:
                if edge == tuple(sorted([elem[0], elem[1]])):
                    midside = elem[3]
                elif edge == tuple(sorted([elem[1], elem[2]])):
                    midside = elem[4]
                else:
                    midside = elem[5]
                assert midside == edge_dict[edge]
            elif edge == tuple(sorted([elem[0], elem[1]])):
                edge_dict[edge] = elem[3]
            elif edge == tuple(sorted([elem[1], elem[2]])):
                edge_dict[edge] = elem[4]
            else:
                edge_dict[edge] = elem[5]

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
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