def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements."""
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords1.shape == (21, 2)
    assert connect1.shape == (4, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    assert np.allclose(coords1[0], [0.0, 0.0])
    assert np.allclose(coords1[2], [0.5, 0.0])
    assert np.allclose(coords1[4], [1.0, 0.0])
    assert np.allclose(coords1[16], [0.0, 1.0])
    assert np.allclose(coords1[18], [0.5, 1.0])
    assert np.allclose(coords1[20], [1.0, 1.0])
    dx = 0.25
    dy = 0.25
    for i in range(coords1.shape[0]):
        x_rem = coords1[i, 0] % dx
        y_rem = coords1[i, 1] % dy
        assert np.isclose(x_rem, 0.0) or np.isclose(x_rem, dx)
        assert np.isclose(y_rem, 0.0) or np.isclose(y_rem, dy)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements."""
    (coords, connect) = fcn(1.0, 2.0, 3.0, 5.0, 2, 3)
    assert np.all(connect >= 0)
    assert np.all(connect < coords.shape[0])
    for elem in connect:
        assert len(np.unique(elem)) == 8
        corners = elem[:4]
        x_corners = coords[corners, 0]
        y_corners = coords[corners, 1]
        area = 0.5 * np.abs(np.dot(x_corners, np.roll(y_corners, 1)) - np.dot(y_corners, np.roll(x_corners, 1)))
        assert area > 0.0
        n5 = coords[elem[4]]
        n1 = coords[elem[0]]
        n2 = coords[elem[1]]
        assert np.allclose(n5, 0.5 * (n1 + n2))
        n6 = coords[elem[5]]
        n3 = coords[elem[2]]
        assert np.allclose(n6, 0.5 * (n2 + n3))
        n7 = coords[elem[6]]
        n4 = coords[elem[3]]
        assert np.allclose(n7, 0.5 * (n3 + n4))
        n8 = coords[elem[7]]
        assert np.allclose(n8, 0.5 * (n4 + n1))
    (nx, ny) = (2, 3)
    for cy in range(ny):
        for cx in range(nx):
            elem_idx = cy * nx + cx
            current_elem = connect[elem_idx]
            if cx < nx - 1:
                right_elem = connect[cy * nx + cx + 1]
                assert current_elem[1] == right_elem[0]
                assert current_elem[2] == right_elem[3]
                assert current_elem[6] == right_elem[7]
            if cy < ny - 1:
                top_elem = connect[(cy + 1) * nx + cx]
                assert current_elem[3] == top_elem[0]
                assert current_elem[2] == top_elem[1]
                assert current_elem[7] == top_elem[5]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation."""
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)