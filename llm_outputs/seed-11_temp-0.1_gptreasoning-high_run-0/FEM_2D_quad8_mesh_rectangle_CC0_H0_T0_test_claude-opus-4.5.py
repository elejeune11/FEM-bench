def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_nnodes = npx * npy - nx * ny
    expected_nelems = nx * ny
    assert coords.shape == (expected_nnodes, 2), f'Expected coords shape {(expected_nnodes, 2)}, got {coords.shape}'
    assert connect.shape == (expected_nelems, 8), f'Expected connect shape {(expected_nelems, 8)}, got {connect.shape}'
    assert coords.dtype == np.float64, f'Expected coords dtype float64, got {coords.dtype}'
    assert connect.dtype == np.int64, f'Expected connect dtype int64, got {connect.dtype}'
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    domain_corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    for corner in domain_corners:
        distances = np.linalg.norm(coords - corner, axis=1)
        assert np.min(distances) < 1e-12, f'Domain corner {corner} not found in coordinates'
    half_dx = dx / 2
    half_dy = dy / 2
    for i in range(coords.shape[0]):
        (x, y) = coords[i]
        kx = (x - xl) / half_dx
        ky = (y - yl) / half_dy
        assert abs(kx - round(kx)) < 1e-12, f'x coordinate {x} not on half-step lattice'
        assert abs(ky - round(ky)) < 1e-12, f'y coordinate {y} not on half-step lattice'
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2), 'Coordinates not deterministic'
    assert np.array_equal(connect, connect2), 'Connectivity not deterministic'

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    nelems = connect.shape[0]
    assert np.all(connect >= 0), 'Connectivity contains negative indices'
    assert np.all(connect < nnodes), f'Connectivity contains indices >= {nnodes}'
    for elem_idx in range(nelems):
        elem_nodes = connect[elem_idx]
        assert len(np.unique(elem_nodes)) == 8, f'Element {elem_idx} has repeated nodes'
    for elem_idx in range(nelems):
        corners = connect[elem_idx, :4]
        corner_coords = coords[corners]
        n = 4
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += corner_coords[i, 0] * corner_coords[j, 1]
            area -= corner_coords[j, 0] * corner_coords[i, 1]
        area /= 2.0
        assert area > 0, f'Element {elem_idx} corners are not counter-clockwise (area={area})'
    midside_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for elem_idx in range(nelems):
        elem_nodes = connect[elem_idx]
        for (mid_idx, (ci, cj)) in enumerate(midside_pairs):
            midside_node = elem_nodes[4 + mid_idx]
            corner_i = elem_nodes[ci]
            corner_j = elem_nodes[cj]
            expected_mid = (coords[corner_i] + coords[corner_j]) / 2.0
            actual_mid = coords[midside_node]
            assert np.allclose(actual_mid, expected_mid, atol=1e-12), f'Element {elem_idx}, midside N{5 + mid_idx} not at average of corners'
    edge_nodes = {}
    edge_definitions = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
    for elem_idx in range(nelems):
        elem_nodes = connect[elem_idx]
        for (c1, mid, c2) in edge_definitions:
            (n1, nm, n2) = (elem_nodes[c1], elem_nodes[mid], elem_nodes[c2])
            edge_key = (min(n1, n2), max(n1, n2))
            if edge_key not in edge_nodes:
                edge_nodes[edge_key] = []
            edge_nodes[edge_key].append((elem_idx, nm))
    for (edge_key, elem_list) in edge_nodes.items():
        if len(elem_list) > 1:
            midside_nodes = [mid for (_, mid) in elem_list]
            assert len(set(midside_nodes)) == 1, f'Shared edge {edge_key} has different midside nodes: {midside_nodes}'

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
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