def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma = fcn(0, 0, 0, 1, 0, 0, None)
    assert np.allclose(gamma[0:3, 0:3], np.eye(3))
    gamma = fcn(0, 0, 0, 0, 1, 0, None)
    assert np.allclose(gamma[0:3, 0:3], np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    gamma = fcn(0, 0, 0, 0, 0, 1, None)
    assert np.allclose(gamma[0:3, 0:3], np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma = fcn(0, 0, 0, 1, 1, 1, np.array([0, 1, 0]))
    assert np.allclose(gamma.shape, (12, 12))
    for i in range(4):
        R = gamma[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3]
        assert np.allclose(R.T @ R, np.eye(3))
    gamma = fcn(0, 0, 0, 1, 0, 0, np.array([0, 0, 1]))
    assert np.allclose(gamma[0:3, 0:3], np.eye(3))

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 1, 1, np.array([0, 2, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 0, np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 1, 1, np.array([0, 0, 1, 1]))