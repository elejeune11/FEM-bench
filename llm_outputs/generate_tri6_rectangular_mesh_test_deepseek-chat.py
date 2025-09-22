def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords.shape == ((2 * 2 + 1) * (2 * 2 + 1), 2)
    assert connect.shape == (2 * 2 * 2, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    npx = 2 * 2 + 1
    npy = 2 * 2 + 1
    assert len(coords) == npx * npy
    assert len(connect) == 2 * 2 * 2
    assert np.min(coords[:, 0]) == 0.0
    assert np.max(coords[:, 0]) == 1.0
    assert np.min(coords[:, 1]) == 0.0
    assert np.max(coords[:, 1]) == 1.0
    x_coords = np.unique(coords[:, 0])
    y_coords = np.unique(coords[:, 1])
    assert np.allclose(np.diff(x_coords), 0.25)
    assert np.allclose(np.diff(y_coords), 0.25)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (coords, connect) = fcn(1.0, 2.0, 3.0, 5.0, 2, 3)
    n_nodes = len(coords)
    assert np.min(connect) >= 0
    assert np.max(connect) < n_nodes
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        corners = elem[:3]
        midsides = elem[3:]
        v1 = coords[corners[1]] - coords[corners[0]]
        v2 = coords[corners[2]] - coords[corners[0]]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross_z > 0
        expected_n4 = (coords[corners[0]] + coords[corners[1]]) / 2
        assert np.allclose(coords[midsides[0]], expected_n4)
        expected_n5 = (coords[corners[1]] + coords[corners[2]]) / 2
        assert np.allclose(coords[midsides[1]], expected_n5)
        expected_n6 = (coords[corners[2]] + coords[corners[0]]) / 2
        assert np.allclose(coords[midsides[2]], expected_n6)
    edge_dict = {}
    for (i, elem) in enumerate(connect):
        edges = [tuple(sorted([elem[0], elem[1]])), tuple(sorted([elem[1], elem[2]])), tuple(sorted([elem[2], elem[0]]))]
        for edge in edges:
            if edge in edge_dict:
                assert edge_dict[edge] == elem[3]
            else:
                edge_dict[edge] = elem[3]

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