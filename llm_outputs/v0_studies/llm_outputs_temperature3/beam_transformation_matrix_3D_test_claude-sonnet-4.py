def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert np.allclose(gamma_x[:3, :3], expected_x)
    gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(gamma_y[:3, :3], expected_y)
    gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(gamma_z[:3, :3], expected_z)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma = fcn(0, 0, 0, 1, 1, 1, None)
    assert gamma.shape == (12, 12)
    R = gamma[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(np.linalg.det(R), 1.0)
    assert np.allclose(gamma[:3, :3], gamma[3:6, 3:6])
    assert np.allclose(gamma[:3, :3], gamma[6:9, 6:9])
    assert np.allclose(gamma[:3, :3], gamma[9:12, 9:12])
    assert np.allclose(gamma[:3, 3:], 0)
    assert np.allclose(gamma[3:6, :3], 0)
    assert np.allclose(gamma[3:6, 6:], 0)
    ref_vec = np.array([0, 1, 0])
    gamma_custom = fcn(0, 0, 0, 1, 0, 0, ref_vec)
    R_custom = gamma_custom[:3, :3]
    local_x = np.array([1, 0, 0])
    assert np.allclose(R_custom[:, 0], local_x)
    expected_local_y = np.cross(ref_vec, local_x)
    expected_local_y = expected_local_y / np.linalg.norm(expected_local_y)
    assert np.allclose(R_custom[:, 1], expected_local_y)

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
        fcn(0, 0, 0, 1, 0, 0, np.array([2, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 0, None)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0, 0]))