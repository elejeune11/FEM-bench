def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.allclose(gamma_x[:3, :3], expected_x)
    gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    assert np.allclose(gamma_y[:3, :3], expected_y)
    gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(gamma_z[:3, :3], expected_z)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma = fcn(0, 0, 0, 1, 1, 0, np.array([0, 0, 1]))
    expected = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0], [-1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 1]])
    assert np.allclose(gamma[:3, :3], expected)
    assert gamma.shape == (12, 12)
    assert np.allclose(gamma[3:6, 3:6], gamma[:3, :3])
    assert np.allclose(gamma[6:9, 6:9], gamma[:3, :3])
    assert np.allclose(gamma[9:12, 9:12], gamma[:3, :3])

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    with pytest.raises(ValueError, match='unit vector'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 1, 1]))
    with pytest.raises(ValueError, match='parallel'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError, match='shape'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))
    with pytest.raises(ValueError, match='zero length'):
        fcn(0, 0, 0, 0, 0, 0, None)