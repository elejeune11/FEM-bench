def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x_block = np.eye(3)
    np.testing.assert_array_almost_equal(gamma_x[:3, :3], expected_x_block)
    gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y_block = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_almost_equal(gamma_y[:3, :3], expected_y_block)
    gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z_block = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_almost_equal(gamma_z[:3, :3], expected_z_block)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma1 = fcn(0, 0, 0, 1, 1, 1, None)
    assert gamma1.shape == (12, 12)
    for i in range(0, 12, 3):
        for j in range(0, 12, 3):
            if i == j:
                block = gamma1[i:i + 3, j:j + 3]
                assert not np.allclose(block, 0)
            else:
                assert np.allclose(gamma1[i:i + 3, j:j + 3], 0)
    ref_vec = np.array([0, 1, 0])
    gamma2 = fcn(0, 0, 0, 1, 0, 0, ref_vec)
    rot_block = gamma2[:3, :3]
    np.testing.assert_array_almost_equal(rot_block.T @ rot_block, np.eye(3))
    np.testing.assert_array_almost_equal(rot_block @ rot_block.T, np.eye(3))
    gamma3 = fcn(0, 0, 0, 2, 0, 0, np.array([0, 0, 1]))
    expected_eye = np.eye(3)
    np.testing.assert_array_almost_equal(gamma3[:3, :3], expected_eye)

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
        fcn(0, 0, 0, 1, 0, 0, np.array([2, 0, 0]))
    with pytest.raises(ValueError, match='parallel'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError, match='shape'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))
    with pytest.raises(ValueError, match='zero length'):
        fcn(0, 0, 0, 0, 0, 0, None)