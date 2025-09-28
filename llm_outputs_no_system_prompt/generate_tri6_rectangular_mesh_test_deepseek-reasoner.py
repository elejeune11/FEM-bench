def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    import numpy as np
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords1.shape == (25, 2)
    assert connect1.shape == (8, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.isclose(coords1[0, 0], 0.0) and np.isclose(coords1[0, 1], 0.0)
    assert np.isclose(coords1[4, 0], 1.0) and np.isclose(coords1[4, 1], 0.0)
    assert np.isclose(coords1[20, 0], 0.0) and np.isclose(coords1[20, 1], 1.0)
    assert np.isclose(coords1[24, 0], 1.0) and np.isclose(coords1[24, 1], 1.0)
    for i in range(5):
        for j in range(5):
            node_id = j * 5 + i
            expected_x = 0.0 + 0.25 * i
            expected_y = 0.0 + 0.25 * j
            assert np.isclose(coords1[node_id, 0], expected_x)
            assert np.isclose(coords1[node_id, 1], expected_y)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    import numpy as np
    (coords, connect) = fcn(1.0, 2.0, 5.0, 6.0, 3, 2)
    n_nodes = coords.shape[0]
    assert np.all(connect >= 0) and np.all(connect < n_nodes)
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        (n1, n2, n3, n4, n5, n6) = elem
        v1 = coords[n2] - coords[n1]
        v2 = coords[n3] - coords[n1]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross_z > 0
        assert np.allclose(coords[n4], 0.5 * (coords[n1] + coords[n2]))
        assert np.allclose(coords[n5], 0.5 * (coords[n2] + coords[n3]))
        assert np.allclose(coords[n6], 0.5 * (coords[n3] + coords[n1]))
    edge_map = {}
    for (elem_idx, elem) in enumerate(connect):
        edges = [tuple(sorted([elem[0], elem[1]])), tuple(sorted([elem[1], elem[2]])), tuple(sorted([elem[2], elem[0]]))]
        for edge in edges:
            if edge not in edge_map:
                edge_map[edge] = []
            edge_map[edge].append(elem_idx)
    for (edge, elements) in edge_map.items():
        if len(elements) == 2:
            (elem1, elem2) = (connect[elements[0]], connect[elements[1]])
            (corner1, corner2) = edge

            def find_midside(elem_nodes, c1, c2):
                if elem_nodes[0] == c1 and elem_nodes[1] == c2 or (elem_nodes[0] == c2 and elem_nodes[1] == c1):
                    return elem_nodes[3]
                elif elem_nodes[1] == c1 and elem_nodes[2] == c2 or (elem_nodes[1] == c2 and elem_nodes[2] == c1):
                    return elem_nodes[4]
                else:
                    return elem_nodes[5]
            midside1 = find_midside(elem1, corner1, corner2)
            midside2 = find_midside(elem2, corner1, corner2)
            assert midside1 == midside2

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    import pytest
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