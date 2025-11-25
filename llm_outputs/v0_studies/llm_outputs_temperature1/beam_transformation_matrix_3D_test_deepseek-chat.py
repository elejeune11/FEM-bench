def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x = np.eye(3)
    assert np.allclose(Gamma_x[:3, :3], expected_x, atol=1e-10)
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(Gamma_y[:3, :3], expected_y, atol=1e-10)
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(Gamma_z[:3, :3], expected_z, atol=1e-10)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct."""
    Gamma1 = fcn(0, 0, 0, 1, 1, 0, np.array([0, 0, 1]))
    assert Gamma1.shape == (12, 12)
    for i in range(0, 12, 3):
        R_block = Gamma1[i:i + 3, i:i + 3]
        identity_check = R_block.T @ R_block
        assert np.allclose(identity_check, np.eye(3), atol=1e-10)
        assert np.allclose(np.linalg.det(R_block), 1.0, atol=1e-10)
    ref_vec = np.array([1, 0, 0])
    Gamma2 = fcn(0, 0, 0, 0, 1, 0, ref_vec)
    R_blocks = [Gamma2[i:i + 3, i:i + 3] for i in range(0, 12, 3)]
    for i in range(1, 4):
        assert np.allclose(R_blocks[0], R_blocks[i], atol=1e-10)
    Gamma3 = fcn(1, 2, 3, 4, 5, 6, np.array([0, 1, 0]))
    assert Gamma3.shape == (12, 12)
    for i in range(4):
        for j in range(4):
            if i != j:
                block = Gamma3[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                assert np.allclose(block, np.zeros((3, 3)), atol=1e-10)

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