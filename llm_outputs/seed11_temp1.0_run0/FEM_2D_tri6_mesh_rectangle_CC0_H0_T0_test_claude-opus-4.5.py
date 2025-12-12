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
            assert np.isclose(coords[node_id, 0], expected_x), f'Node {node_id} x mismatch'
            assert np.isclose(coords[node_id, 1], expected_y), f'Node {node_id} y mismatch'
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2), 'Coords not deterministic'
    assert np.array_equal(connect, connect2), 'Connect not deterministic'

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
    assert np.all(connect >= 0), 'Connectivity contains negative indices'
    assert np.all(connect < nnodes), f'Connectivity contains indices >= {nnodes}'
    for elem_idx in range(nelems):
        elem_nodes = connect[elem_idx]
        assert len(np.unique(elem_nodes)) == 6, f'Element {elem_idx} does not have 6 unique nodes'
    for elem_idx in range(nelems):
        (n1, n2, n3) = connect[elem_idx, :3]
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        signed_area = 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        assert signed_area > 0, f'Element {elem_idx} corners are not CCW (signed area = {signed_area})'
    for elem_idx in range(nelems):
        (n1, n2, n3, n4, n5, n6) = connect[elem_idx]
        expected_n4 = 0.5 * (coords[n1] + coords[n2])
        assert np.allclose(coords[n4], expected_n4), f'Element {elem_idx} N4 mismatch'
        expected_n5 = 0.5 * (coords[n2] + coords[n3])
        assert np.allclose(coords[n5], expected_n5), f'Element {elem_idx} N5 mismatch'
        expected_n6 = 0.5 * (coords[n3] + coords[n1])
        assert np.allclose(coords[n6], expected_n6), f'Element {elem_idx} N6 mismatch'
    edge_map = {}
    for elem_idx in range(nelems):
        (n1, n2, n3, n4, n5, n6) = connect[elem_idx]
        edges = [(min(n1, n2), max(n1, n2), n4), (min(n2, n3), max(n2, n3), n5), (min(n3, n1), max(n3, n1), n6)]
        for (edge_key_a, edge_key_b, midnode) in edges:
            edge_key = (edge_key_a, edge_key_b)
            if edge_key in edge_map:
                prev_midnode = edge_map[edge_key]
                assert midnode == prev_midnode, f'Shared edge {edge_key} has different midnodes: {prev_midnode} vs {midnode}'
            else:
                edge_map[edge_key] = midnode

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