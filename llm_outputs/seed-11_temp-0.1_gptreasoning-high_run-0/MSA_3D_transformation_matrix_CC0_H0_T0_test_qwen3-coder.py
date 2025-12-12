def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    np.testing.assert_array_almost_equal(Gamma_x[:3, :3], expected_x)
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_almost_equal(Gamma_y[:3, :3], expected_y)
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    np.testing.assert_array_almost_equal(Gamma_z[:3, :3], expected_z)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the transformation is correct."""
    Gamma = fcn(0, 0, 0, 1, 1, 1, None)
    R = Gamma[:3, :3]
    np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0), 'Determinant of rotation matrix should be 1'
    for i in range(4):
        np.testing.assert_array_equal(Gamma[:3, :3], Gamma[3 * i:3 * i + 3, 3 * i:3 * i + 3])
    ref_vec = np.array([1, 0, 0]) / np.sqrt(1)
    Gamma_custom = fcn(0, 0, 0, 0, 1, 0, ref_vec)
    R_custom = Gamma_custom[:3, :3]
    np.testing.assert_array_almost_equal(R_custom @ R_custom.T, np.eye(3))
    assert np.isclose(np.linalg.det(R_custom), 1.0)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also verifies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError"""
    with pytest.raises(ValueError, match='reference_vector'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 1, 0]))
    with pytest.raises(ValueError, match='reference_vector'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))
    with pytest.raises(ValueError, match='parallel'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError, match='zero length'):
        fcn(0, 0, 0, 0, 0, 0, None)