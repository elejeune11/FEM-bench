def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, None)
    expected_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(gamma_x[:3, :3], expected_x, atol=1e-12)
    gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, None)
    expected_y = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(gamma_y[:3, :3], expected_y, atol=1e-12)
    gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None)
    expected_z = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(gamma_z[:3, :3], expected_z, atol=1e-12)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma = fcn(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, None)
    assert gamma.shape == (12, 12)
    R = gamma[:3, :3]
    assert np.allclose(gamma[3:6, 3:6], R, atol=1e-12)
    assert np.allclose(gamma[6:9, 6:9], R, atol=1e-12)
    assert np.allclose(gamma[9:12, 9:12], R, atol=1e-12)
    for i in range(4):
        for j in range(4):
            if i != j:
                block = gamma[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
                assert np.allclose(block, np.zeros((3, 3)), atol=1e-12)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)
    ref_vec = np.array([1.0, 0.0, 0.0])
    gamma_custom = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ref_vec)
    R_custom = gamma_custom[:3, :3]
    assert np.allclose(R_custom[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(R_custom[:, 1], [0.0, 0.0, -1.0], atol=1e-12)
    assert np.allclose(R_custom[:, 2], [1.0, 0.0, 0.0], atol=1e-12)

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
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([2.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, None)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 0.0]))