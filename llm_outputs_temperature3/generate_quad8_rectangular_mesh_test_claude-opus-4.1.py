def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    import numpy as np
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_nelems = nx * ny
    assert coords.shape == (expected_nnodes, 2)
    assert connect.shape == (expected_nelems, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    corner_coords = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    for corner in corner_coords:
        assert np.any(np.all(np.isclose(coords, corner), axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    for coord in coords:
        x_steps = (coord[0] - xl) / half_dx
        y_steps = (coord[1] - yl) / half_dy
        assert np.isclose(x_steps, np.round(x_steps))
        assert np.isclose(y_steps, np.round(y_steps))
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    import numpy as np
    (xl, yl, xh, yh) = (-1.0, -2.0, 3.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    for elem in connect:
        assert np.all(elem >= 0)
        assert np.all(elem < nnodes)
        assert len(np.unique(elem)) == 8
    for elem in connect:
        corners = coords[elem[:4]]
        area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            area += corners[i, 0] * corners[j, 1] - corners[j, 0] * corners[i, 1]
        area *= 0.5
        assert area > 0
        midside_pairs = [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]
        for (c1, c2, m) in midside_pairs:
            expected_mid = 0.5 * (coords[elem[c1]] + coords[elem[c2]])
            assert np.allclose(coords[elem[m]], expected_mid)
    edge_map = {}
    for (elem_idx, elem) in enumerate(connect):
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[3]), (elem[3], elem[0])]
        for edge in edges:
            sorted_edge = tuple(sorted(edge))
            if sorted_edge in edge_map:
                edge_map[sorted_edge].append(elem_idx)
            else:
                edge_map[sorted_edge] = [elem_idx]
    for (edge, elems) in edge_map.items():
        if len(elems) > 1:
            for elem_idx in elems[1:]:
                assert edge[0] in connect[elems[0]]
                assert edge[1] in connect[elems[0]]
                assert edge[0] in connect[elem_idx]
                assert edge[1] in connect[elem_idx]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
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