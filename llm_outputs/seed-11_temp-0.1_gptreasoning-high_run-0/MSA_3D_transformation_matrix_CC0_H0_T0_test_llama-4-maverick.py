def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes."""
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (1.0, 0.0, 0.0)
    gamma = fcn(x1, y1, z1, x2, y2, z2, None)
    expected_dcm = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.allclose(gamma[:3, :3], expected_dcm)
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (0.0, 1.0, 0.0)
    gamma = fcn(x1, y1, z1, x2, y2, z2, None)
    expected_dcm = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(gamma[:3, :3], expected_dcm)
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (0.0, 0.0, 1.0)
    gamma = fcn(x1, y1, z1, x2, y2, z2, None)
    expected_dcm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(gamma[:3, :3], expected_dcm)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations."""
    (x1, y1, z1) = (1.0, 2.0, 3.0)
    (x2, y2, z2) = (4.0, 5.0, 6.0)
    reference_vector = np.array([0.0, 0.0, 1.0])
    gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    assert gamma.shape == (12, 12)
    assert np.allclose(gamma[:3, :3], gamma[3:6, 3:6])
    assert np.allclose(gamma[:3, :3], gamma[6:9, 6:9])
    assert np.allclose(gamma[:3, :3], gamma[9:, 9:])
    assert np.allclose(gamma.T @ gamma, np.eye(12))

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors."""
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (1.0, 0.0, 0.0)
    reference_vector = np.array([0.0, 0.0, 2.0])
    with pytest.raises(ValueError):
        fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    reference_vector = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    (x2, y2, z2) = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        fcn(x1, y1, z1, x2, y2, z2, None)
    reference_vector = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, reference_vector)