def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain."""
    (coords, connect) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords.shape == (25, 2)
    assert connect.shape == (8, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    (npx, npy) = (2 * 2 + 1, 2 * 2 + 1)
    expected_nodes = npx * npy
    expected_elements = 2 * 2 * 2
    assert len(coords) == expected_nodes
    assert len(connect) == expected_elements
    assert np.allclose(coords[0], [0.0, 0.0])
    assert np.allclose(coords[4], [1.0, 0.0])
    assert np.allclose(coords[20], [0.0, 1.0])
    assert np.allclose(coords[24], [1.0, 1.0])
    (dx, dy) = (0.5, 0.5)
    for iy in range(npy):
        for ix in range(npx):
            node_id = iy * npx + ix
            expected_x = 0.0 + 0.5 * dx * ix
            expected_y = 0.0 + 0.5 * dy * iy
            assert np.allclose(coords[node_id], [expected_x, expected_y])
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain."""
    (coords, connect) = fcn(0.0, 0.0, 2.0, 1.0, 3, 2)
    n_nodes = len(coords)
    for elem in connect:
        assert len(np.unique(elem)) == 6
        assert np.all(elem >= 0) and np.all(elem < n_nodes)
        corners = elem[:3]
        v1 = coords[corners[1]] - coords[corners[0]]
        v2 = coords[corners[2]] - coords[corners[0]]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross_z > 0
        midsides = elem[3:]
        expected_mid1 = (coords[corners[0]] + coords[corners[1]]) / 2
        expected_mid2 = (coords[corners[1]] + coords[corners[2]]) / 2
        expected_mid3 = (coords[corners[2]] + coords[corners[0]]) / 2
        assert np.allclose(coords[midsides[0]], expected_mid1)
        assert np.allclose(coords[midsides[1]], expected_mid2)
        assert np.allclose(coords[midsides[2]], expected_mid3)
    edge_map = {}
    for (elem_idx, elem) in enumerate(connect):
        edges = [(elem[0], elem[1], elem[3]), (elem[1], elem[2], elem[4]), (elem[2], elem[0], elem[5])]
        for edge in edges:
            sorted_corners = tuple(sorted(edge[:2]))
            if sorted_corners in edge_map:
                assert edge_map[sorted_corners][2] == edge[2]
                edge_map[sorted_corners] = (edge_map[sorted_corners][0], edge_map[sorted_corners][1] + 1, edge[2])
            else:
                edge_map[sorted_corners] = (elem_idx, 1, edge[2])
    for edge_info in edge_map.values():
        assert edge_info[1] == 2

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs."""
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 2, 2)