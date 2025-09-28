def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
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
    assert np.allclose(np.diff(x_coords), 0.25)
    assert np.allclose(np.diff(y_coords), 0.25)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 4.0, 5.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    assert np.min(connect) >= 0
    assert np.max(connect) < n_nodes
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        corners = elem[:3]
        midsides = elem[3:]
        corner_coords = coords[corners]
        vec1 = corner_coords[1] - corner_coords[0]
        vec2 = corner_coords[2] - corner_coords[0]
        cross_z = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        assert cross_z > 0
        expected_midsides = [(corner_coords[0] + corner_coords[1]) / 2, (corner_coords[1] + corner_coords[2]) / 2, (corner_coords[2] + corner_coords[0]) / 2]
        for (i, expected) in enumerate(expected_midsides):
            actual = coords[midsides[i]]
            assert np.allclose(actual, expected)
    edge_dict = {}
    for (elem_idx, elem) in enumerate(connect):
        edges = [(elem[0], elem[1], elem[3]), (elem[1], elem[2], elem[4]), (elem[2], elem[0], elem[5])]
        for (n1, n2, mid) in edges:
            edge_key = tuple(sorted([n1, n2]))
            if edge_key in edge_dict:
                assert edge_dict[edge_key] == mid
            else:
                edge_dict[edge_key] = mid

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)