def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    Test cases:
    """
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (1.0, 0.0, 0.0)
    reference_vector = np.array([0.0, 0.0, 1.0])
    Gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    expected_local_x = np.array([1.0, 0.0, 0.0])
    expected_local_y = np.array([0.0, 1.0, 0.0])
    expected_local_z = np.array([0.0, 0.0, 1.0])
    assert np.allclose(Gamma[:3, :3], np.column_stack((expected_local_x, expected_local_y, expected_local_z)))
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (0.0, 1.0, 0.0)
    reference_vector = np.array([0.0, 0.0, 1.0])
    Gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    expected_local_x = np.array([0.0, 1.0, 0.0])
    expected_local_y = np.array([1.0, 0.0, 0.0])
    expected_local_z = np.array([0.0, 0.0, 1.0])
    assert np.allclose(Gamma[:3, :3], np.column_stack((expected_local_x, expected_local_y, expected_local_z)))
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (0.0, 0.0, 1.0)
    reference_vector = np.array([0.0, 1.0, 0.0])
    Gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    expected_local_x = np.array([0.0, 0.0, 1.0])
    expected_local_y = np.array([1.0, 0.0, 0.0])
    expected_local_z = np.array([0.0, 1.0, 0.0])
    assert np.allclose(Gamma[:3, :3], np.column_stack((expected_local_x, expected_local_y, expected_local_z)))

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the transformation is correct.
    """
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (1.0, 1.0, 1.0)
    reference_vector = np.array([0.0, 0.0, 1.0])
    Gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    assert Gamma.shape == (12, 12)
    assert np.allclose(Gamma.T @ Gamma, np.eye(12))
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (1.0, 0.0, 0.0)
    reference_vector = np.array([0.0, 0.0, 1.0])
    Gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector)
    assert Gamma.shape == (12, 12)
    assert np.allclose(Gamma.T @ Gamma, np.eye(12))

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    (x1, y1, z1) = (0.0, 0.0, 0.0)
    (x2, y2, z2) = (1.0, 0.0, 0.0)
    reference_vector = np.array([0.0, 0.0, 2.0])
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