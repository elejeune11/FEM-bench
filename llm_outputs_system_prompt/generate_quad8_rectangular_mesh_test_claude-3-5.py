def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements."""
    import numpy as np
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    expected_nodes = (2 * 2 + 1) * (2 * 2 + 1) - 2 * 2
    expected_elements = 2 * 2
    assert coords1.shape == (expected_nodes, 2)
    assert connect1.shape == (expected_elements, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    for corner in corners:
        assert any(np.all(np.abs(coords1 - corner) < 1e-14, axis=1))
    x_coords = np.unique(coords1[:, 0])
    y_coords = np.unique(coords1[:, 1])
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    assert np.allclose(dx, dx[0])
    assert np.allclose(dy, dy[0])
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements."""
    import numpy as np
    (coords, connect) = fcn(-1.0, -2.0, 2.0, 1.0, 2, 2)
    assert np.all(connect >= 0)
    assert np.all(connect < len(coords))
    for elem in connect:
        corners = coords[elem[:4]]
        area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            area += corners[i, 0] * corners[j, 1] - corners[j, 0] * corners[i, 1]
        assert area > 0
    for elem in connect:
        for i in range(4):
            mid_idx = elem[4 + i]
            c1_idx = elem[i]
            c2_idx = elem[(i + 1) % 4]
            expected_mid = 0.5 * (coords[c1_idx] + coords[c2_idx])
            assert np.allclose(coords[mid_idx], expected_mid)
    edge_nodes = {}
    for (i, elem) in enumerate(connect):
        edges = [(elem[j], elem[(j + 1) % 4]) for j in range(4)]
        for edge in edges:
            edge = tuple(sorted(edge))
            if edge in edge_nodes:
                edge_nodes[edge].append(i)
            else:
                edge_nodes[edge] = [i]
    for (edge, elements) in edge_nodes.items():
        if len(elements) == 2:
            (e1, e2) = elements
            assert edge[0] in connect[e1] and edge[0] in connect[e2]
            assert edge[1] in connect[e1] and edge[1] in connect[e2]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation."""
    import pytest
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 2, 2)