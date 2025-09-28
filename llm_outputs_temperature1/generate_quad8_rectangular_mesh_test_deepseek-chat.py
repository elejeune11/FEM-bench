def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    np.testing.assert_array_equal(coords1, coords2)
    np.testing.assert_array_equal(connect1, connect2)
    assert coords1.shape == (21, 2)
    assert connect1.shape == (4, 8)
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

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (coords, connect) = fcn(1.0, 2.0, 3.0, 5.0, 2, 3)
    n_nodes = coords.shape[0]
    n_elems = connect.shape[0]
    assert np.min(connect) >= 0
    assert np.max(connect) < n_nodes
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        corners = elem[:4]
        corner_coords = coords[corners]
        (x, y) = (corner_coords[:, 0], corner_coords[:, 1])
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        assert area > 0
        (N1, N2, N3, N4, N5, N6, N7, N8) = elem
        np.testing.assert_allclose(coords[N5], 0.5 * (coords[N1] + coords[N2]))
        np.testing.assert_allclose(coords[N6], 0.5 * (coords[N2] + coords[N3]))
        np.testing.assert_allclose(coords[N7], 0.5 * (coords[N3] + coords[N4]))
        np.testing.assert_allclose(coords[N8], 0.5 * (coords[N4] + coords[N1]))
    (nx, ny) = (2, 3)
    for i in range(ny):
        for j in range(nx):
            elem_idx = i * nx + j
            current_elem = connect[elem_idx]
            if j < nx - 1:
                right_elem = connect[elem_idx + 1]
                assert current_elem[1] == right_elem[3]
                assert current_elem[2] == right_elem[0]
                assert current_elem[5] == right_elem[7]
            if i < ny - 1:
                top_elem = connect[elem_idx + nx]
                assert current_elem[2] == top_elem[0]
                assert current_elem[3] == top_elem[1]
                assert current_elem[6] == top_elem[4]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
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