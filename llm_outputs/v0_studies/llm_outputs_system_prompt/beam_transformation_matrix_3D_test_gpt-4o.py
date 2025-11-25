def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes."""
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (1.0, 0.0, 0.0)
    reference_vector = np.array([0.0, 0.0, 1.0])
    Gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    local_x = np.array([1.0, 0.0, 0.0])
    local_y = np.array([0.0, 1.0, 0.0])
    local_z = np.array([0.0, 0.0, 1.0])
    assert np.allclose(Gamma[:3, :3], np.column_stack((local_x, local_y, local_z)))
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (0.0, 1.0, 0.0)
    reference_vector = np.array([0.0, 0.0, 1.0])
    Gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    local_x = np.array([0.0, 1.0, 0.0])
    local_y = np.array([-1.0, 0.0, 0.0])
    local_z = np.array([0.0, 0.0, 1.0])
    assert np.allclose(Gamma[:3, :3], np.column_stack((local_x, local_y, local_z)))
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (0.0, 0.0, 1.0)
    reference_vector = np.array([0.0, 1.0, 0.0])
    Gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    local_x = np.array([0.0, 0.0, 1.0])
    local_y = np.array([0.0, 1.0, 0.0])
    local_z = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(Gamma[:3, :3], np.column_stack((local_x, local_y, local_z)))

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations."""
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (1.0, 1.0, 0.0)
    reference_vector = np.array([0.0, 0.0, 1.0])
    Gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    assert np.allclose(Gamma[:3, :3].T @ Gamma[:3, :3], np.eye(3))
    assert np.isclose(np.linalg.det(Gamma[:3, :3]), 1.0)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors."""
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (1.0, 0.0, 0.0)
    reference_vector = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    reference_vector = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (0.0, 0.0, 0.0)
    reference_vector = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        fcn(x1, y1, z1, x2, y2, z2, reference_vector)