def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_nnodes = npx * npy
    expected_nelems = 2 * nx * ny
    assert coords.shape == (expected_nnodes, 2), f'Expected coords shape {(expected_nnodes, 2)}, got {coords.shape}'
    assert connect.shape == (expected_nelems, 6), f'Expected connect shape {(expected_nelems, 6)}, got {connect.shape}'
    assert coords.dtype == np.float64, f'Expected coords dtype float64, got {coords.dtype}'
    assert connect.dtype == np.int64, f'Expected connect dtype int64, got {connect.dtype}'
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    assert np.isclose(x_coords.min(), xl), f'Min x should be {xl}, got {x_coords.min()}'
    assert np.isclose(x_coords.max(), xh), f'Max x should be {xh}, got {x_coords.max()}'
    assert np.isclose(y_coords.min(), yl), f'Min y should be {yl}, got {y_coords.min()}'
    assert np.isclose(y_coords.max(), yh), f'Max y should be {yh}, got {y_coords.max()}'
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    for iy in range(npy):
        for ix in range(npx):
            node_id = iy * npx + ix
            expected_x = xl + 0.5 * dx * ix
            expected_y = yl + 0.5 * dy * iy
            assert np.isclose(coords[node_id, 0], expected_x), f'Node {node_id} x: expected {expected_x}, got {coords[node_id, 0]}'
            assert np.isclose(coords[node_id, 1], expected_y), f'Node {node_id} y: expected {expected_y}, got {coords[node_id, 1]}'
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2), 'Repeated calls should yield identical coords'
    assert np.array_equal(connect, connect2), 'Repeated calls should yield identical connect'

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    nnodes = npx * npy
    nelems = 2 * nx * ny
    assert np.all(connect >= 0), 'All connectivity indices should be non-negative'
    assert np.all(connect < nnodes), f'All connectivity indices should be less than {nnodes}'
    for elem_idx in range(nelems):
        elem_nodes = connect[elem_idx]
        assert len(np.unique(elem_nodes)) == 6, f'Element {elem_idx} should have 6 unique nodes'
    for elem_idx in range(nelems):
        (n1, n2, n3, n4, n5, n6) = connect[elem_idx]
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross > 0, f'Element {elem_idx} corners should be CCW (cross product = {cross})'
        expected_n4 = (p1 + p2) / 2
        assert np.allclose(coords[n4], expected_n4), f'Element {elem_idx} N4 should be midpoint of N1-N2'
        expected_n5 = (p2 + p3) / 2
        assert np.allclose(coords[n5], expected_n5), f'Element {elem_idx} N5 should be midpoint of N2-N3'
        expected_n6 = (p3 + p1) / 2
        assert np.allclose(coords[n6], expected_n6), f'Element {elem_idx} N6 should be midpoint of N3-N1'
    edge_to_elems = {}
    for elem_idx in range(nelems):
        (n1, n2, n3, n4, n5, n6) = connect[elem_idx]
        edges = [tuple(sorted([n1, n2])), tuple(sorted([n2, n3])), tuple(sorted([n3, n1]))]
        midsides = [n4, n5, n6]
        for (i, edge) in enumerate(edges):
            if edge not in edge_to_elems:
                edge_to_elems[edge] = []
            edge_to_elems[edge].append((elem_idx, midsides[i]))
    for (edge, elem_info) in edge_to_elems.items():
        if len(elem_info) > 1:
            midsides = [info[1] for info in elem_info]
            assert len(set(midsides)) == 1, f'Shared edge {edge} should have same midside node, got {midsides}'

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