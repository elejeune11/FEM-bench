def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements."""
    (coords1, connect1) = fcn(0, 0, 2, 2, 2, 2)
    (coords2, connect2) = fcn(0, 0, 2, 2, 2, 2)
    assert coords1.shape == (21, 2)
    assert connect1.shape == (4, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.allclose(coords1, coords2)
    assert np.allclose(connect1, connect2)
    assert np.all(coords1[0] == [0, 0])
    assert np.all(coords1[-1] == [2, 2])

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements."""
    (coords, connect) = fcn(0, 0, 3, 2, 3, 2)
    assert np.all(connect >= 0)
    assert np.all(connect < coords.shape[0])

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation."""
    with pytest.raises(ValueError):
        fcn(0, 0, 2, 2, 0, 2)
    with pytest.raises(ValueError):
        fcn(0, 0, 2, 2, 2, 0)
    with pytest.raises(ValueError):
        fcn(2, 0, 0, 2, 2, 2)
    with pytest.raises(ValueError):
        fcn(0, 2, 2, 0, 2, 2)