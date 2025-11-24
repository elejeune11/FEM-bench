def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x = np.eye(3)
    assert np.allclose(gamma_x[:3, :3], expected_x)
    gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(gamma_y[:3, :3], expected_y)
    gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(gamma_z[:3, :3], expected_z)
    for gamma in [gamma_x, gamma_y, gamma_z]:
        assert gamma.shape == (12, 12)
        zero_blocks = [(0, 3), (0, 6), (0, 9), (3, 0), (3, 6), (3, 9), (6, 0), (6, 3), (6, 9), (9, 0), (9, 3), (9, 6)]
        for (i, j) in zero_blocks:
            assert np.allclose(gamma[i:i + 3, j:j + 3], np.zeros((3, 3)))
        R = gamma[:3, :3]
        assert np.allclose(gamma[3:6, 3:6], R)
        assert np.allclose(gamma[6:9, 6:9], R)
        assert np.allclose(gamma[9:12, 9:12], R)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct."""
    gamma1 = fcn(0, 0, 0, 1, 0, 0, np.array([0, 1, 0]))
    expected1 = np.eye(3)
    assert np.allclose(gamma1[:3, :3], expected1)
    gamma2 = fcn(0, 0, 0, 1, 1, 1, np.array([0, 0, 1]))
    length = np.sqrt(3)
    expected_local_x = np.array([1, 1, 1]) / length
    expected_local_y = np.cross(np.array([0, 0, 1]), expected_local_x)
    expected_local_y /= np.linalg.norm(expected_local_y)
    expected_local_z = np.cross(expected_local_x, expected_local_y)
    expected_R = np.column_stack([expected_local_x, expected_local_y, expected_local_z])
    assert np.allclose(gamma2[:3, :3], expected_R)
    for gamma in [gamma1, gamma2]:
        R = gamma[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3))
        assert np.allclose(np.linalg.det(R), 1.0)
    K_local = np.random.rand(12, 12)
    K_local = (K_local + K_local.T) / 2
    gamma3 = fcn(0, 0, 0, 2, 0, 0, None)
    K_global = gamma3.T @ K_local @ gamma3
    assert K_global.shape == (12, 12)
    assert np.allclose(K_global, K_global.T)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError"""
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([2, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 0, None)