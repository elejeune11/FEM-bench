def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x_block = np.eye(3)
    assert np.allclose(Gamma_x[:3, :3], expected_x_block)
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y_block = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(Gamma_y[:3, :3], expected_y_block)
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z_block = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(Gamma_z[:3, :3], expected_z_block)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct."""
    Gamma1 = fcn(0, 0, 0, 1, 1, 1, np.array([0, 0, 1]))
    assert Gamma1.shape == (12, 12)
    rotation_block = Gamma1[:3, :3]
    for i in range(3, 12, 3):
        assert np.allclose(Gamma1[i:i + 3, i:i + 3], rotation_block)
    assert np.allclose(rotation_block @ rotation_block.T, np.eye(3))
    assert np.allclose(rotation_block.T @ rotation_block, np.eye(3))
    ref_vec = np.array([1, 0, 0])
    ref_vec = ref_vec / norm(ref_vec)
    Gamma2 = fcn(0, 0, 0, 0, 1, 0, ref_vec)
    beam_dir = np.array([0, 1, 0])
    beam_dir = beam_dir / norm(beam_dir)
    local_x = Gamma2[:3, 0]
    assert np.allclose(local_x, beam_dir)

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
    with pytest.raises(ValueError, match='zero length'):
        fcn(0, 0, 0, 0, 0, 0, None)
    with pytest.raises(ValueError, match='shape'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))