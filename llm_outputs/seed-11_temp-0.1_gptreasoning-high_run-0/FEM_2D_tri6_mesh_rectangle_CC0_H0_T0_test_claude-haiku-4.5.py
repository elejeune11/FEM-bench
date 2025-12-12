def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1)
    expected_nelems = 2 * nx * ny
    assert coords1.shape == (expected_nnodes, 2)
    assert connect1.shape == (expected_nelems, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    np.testing.assert_array_equal(coords1, coords2)
    np.testing.assert_array_equal(connect1, connect2)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    assert coords1[0, 0] == xl
    assert coords1[0, 1] == yl
    assert coords1[2 * nx, 0] == xh
    assert coords1[2 * nx, 1] == yl
    npx = 2 * nx + 1
    top_left_id = 2 * ny * npx
    assert coords1[top_left_id, 0] == xl
    assert coords1[top_left_id, 1] == yh
    top_right_id = 2 * ny * npx + 2 * nx
    assert coords1[top_right_id, 0] == xh
    assert coords1[top_right_id, 1] == yh
    for iy in range(2 * ny + 1):
        for ix in range(2 * nx + 1):
            node_id = iy * npx + ix
            expected_x = xl + 0.5 * dx * ix
            expected_y = yl + 0.5 * dy * iy
            assert np.isclose(coords1[node_id, 0], expected_x)
            assert np.isclose(coords1[node_id, 1], expected_y)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 5.0, 7.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = (2 * nx + 1) * (2 * ny + 1)
    nelems = 2 * nx * ny
    assert np.all(connect >= 0)
    assert np.all(connect < nnodes)
    for elem_id in range(nelems):
        nodes = connect[elem_id]
        assert len(np.unique(nodes)) == 6, f'Element {elem_id} has duplicate nodes'
    for elem_id in range(nelems):
        (n1, n2, n3, n4, n5, n6) = connect[elem_id]
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross > 0, f'Element {elem_id} is not CCW'
        p4 = coords[n4]
        p5 = coords[n5]
        p6 = coords[n6]
        expected_p4 = (p1 + p2) / 2
        expected_p5 = (p2 + p3) / 2
        expected_p6 = (p3 + p1) / 2
        np.testing.assert_allclose(p4, expected_p4, rtol=1e-14)
        np.testing.assert_allclose(p5, expected_p5, rtol=1e-14)
        np.testing.assert_allclose(p6, expected_p6, rtol=1e-14)
    edge_to_elems = {}
    for elem_id in range(nelems):
        (n1, n2, n3, n4, n5, n6) = connect[elem_id]
        edges = [(min(n1, n2), max(n1, n2)), (min(n2, n3), max(n2, n3)), (min(n3, n1), max(n3, n1))]
        for edge in edges:
            if edge not in edge_to_elems:
                edge_to_elems[edge] = []
            edge_to_elems[edge].append(elem_id)
    for (edge, elems) in edge_to_elems.items():
        if len(elems) == 2:
            (elem1, elem2) = elems
            nodes1 = set(connect[elem1])
            nodes2 = set(connect[elem2])
            assert edge[0] in nodes1 and edge[1] in nodes1
            assert edge[0] in nodes2 and edge[1] in nodes2

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, -1, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 1, 0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 1, -1)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 1.0, yh, 1, 1)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 0.0, yh, 1, 1)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 0.0, 1, 1)