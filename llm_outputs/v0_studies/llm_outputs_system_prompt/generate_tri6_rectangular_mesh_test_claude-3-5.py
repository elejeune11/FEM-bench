def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain."""
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords1.shape == (25, 2)
    assert connect1.shape == (8, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.allclose(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    assert np.allclose(coords1[0], [0.0, 0.0])
    assert np.allclose(coords1[4], [1.0, 0.0])
    assert np.allclose(coords1[20], [0.0, 1.0])
    assert np.allclose(coords1[24], [1.0, 1.0])
    dx = 0.25
    dy = 0.25
    for iy in range(5):
        for ix in range(5):
            node_id = iy * 5 + ix
            assert np.allclose(coords1[node_id], [dx * ix, dy * iy])

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain."""
    (coords, connect) = fcn(-1.0, 2.0, 2.0, 4.0, 1, 1)
    assert np.all(connect >= 0)
    assert np.all(connect < len(coords))
    for elem in connect:
        (n1, n2, n3, n4, n5, n6) = elem
        (p1, p2, p3) = (coords[n1], coords[n2], coords[n3])
        v1 = p2 - p1
        v2 = p3 - p1
        assert np.cross(v1, v2) > 0
        assert np.allclose(coords[n4], 0.5 * (coords[n1] + coords[n2]))
        assert np.allclose(coords[n5], 0.5 * (coords[n2] + coords[n3]))
        assert np.allclose(coords[n6], 0.5 * (coords[n3] + coords[n1]))
    edge_nodes = {}
    for (i, elem) in enumerate(connect):
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        mids = [elem[3], elem[4], elem[5]]
        for ((n1, n2), mid) in zip(edges, mids):
            key = tuple(sorted([n1, n2]))
            if key in edge_nodes:
                assert edge_nodes[key] == mid
            else:
                edge_nodes[key] = mid

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs."""
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)