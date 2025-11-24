def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x = np.eye(3)
    np.testing.assert_array_almost_equal(Gamma_x[:3, :3], expected_x)
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_almost_equal(Gamma_y[:3, :3], expected_y)
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_almost_equal(Gamma_z[:3, :3], expected_z)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct."""
    Gamma = fcn(0, 0, 0, 1, 0, 0, np.array([0, 1, 0]))
    expected = np.eye(3)
    np.testing.assert_array_almost_equal(Gamma[:3, :3], expected)
    Gamma_45 = fcn(0, 0, 0, 1, 1, 0, np.array([0, 0, 1]))
    sqrt2 = np.sqrt(2)
    expected_45 = np.array([[1 / sqrt2, 1 / sqrt2, 0], [-1 / sqrt2, 1 / sqrt2, 0], [0, 0, 1]])
    np.testing.assert_array_almost_equal(Gamma_45[:3, :3], expected_45)
    beam_coords = [(0, 0, 0, 1, 1, 1), (0, 0, 0, 2, -1, 0), (1, 2, 3, 4, 5, 6)]
    ref_vectors = [np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])]
    for (coords, ref_vec) in zip(beam_coords, ref_vectors):
        Gamma = fcn(*coords, ref_vec)
        R = Gamma[:3, :3]
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3))
        assert np.allclose(np.linalg.det(R), 1.0)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError"""
    with pytest.raises(ValueError, match='unit vector'):
        fcn(0, 0, 0, 1, 0, 0, np.array([2, 0, 0]))
    with pytest.raises(ValueError, match='parallel'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError, match='shape'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))
    with pytest.raises(ValueError, match='zero length'):
        fcn(0, 0, 0, 0, 0, 0, np.array([0, 1, 0]))