def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    R_expected_x = np.eye(3)
    assert np.allclose(Gamma_x[:3, :3], R_expected_x, atol=1e-10)
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    R_expected_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(Gamma_y[:3, :3], R_expected_y, atol=1e-10)
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    R_expected_z = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(Gamma_z[:3, :3], R_expected_z, atol=1e-10)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct."""
    Gamma1 = fcn(0, 0, 0, 1, 1, 0, np.array([0, 0, 1]))
    assert Gamma1.shape == (12, 12)
    R_block = Gamma1[:3, :3]
    assert np.allclose(Gamma1[3:6, 3:6], R_block, atol=1e-10)
    assert np.allclose(Gamma1[6:9, 6:9], R_block, atol=1e-10)
    assert np.allclose(Gamma1[9:12, 9:12], R_block, atol=1e-10)
    assert np.allclose(R_block.T @ R_block, np.eye(3), atol=1e-10)
    ref_vec = np.array([1, 0, 0])
    Gamma2 = fcn(0, 0, 0, 0, 1, 0, ref_vec)
    R_block2 = Gamma2[:3, :3]
    assert np.allclose(R_block2.T @ R_block2, np.eye(3), atol=1e-10)
    assert abs(np.linalg.det(R_block2) - 1.0) < 1e-10

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
        fcn(0, 0, 0, 0, 0, 0, None)