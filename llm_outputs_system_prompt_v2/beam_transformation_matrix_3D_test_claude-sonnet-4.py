def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, None)
    R_x = Gamma_x[:3, :3]
    expected_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    assert np.allclose(R_x, expected_x)
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, None)
    R_y = Gamma_y[:3, :3]
    expected_y = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert np.allclose(R_y, expected_y)
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None)
    R_z = Gamma_z[:3, :3]
    expected_z = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(R_z, expected_z)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    Gamma = fcn(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, None)
    assert Gamma.shape == (12, 12)
    R = Gamma[:3, :3]
    assert np.allclose(Gamma[3:6, 3:6], R)
    assert np.allclose(Gamma[6:9, 6:9], R)
    assert np.allclose(Gamma[9:12, 9:12], R)
    assert np.allclose(Gamma[:3, 3:], 0.0)
    assert np.allclose(Gamma[3:6, :3], 0.0)
    assert np.allclose(Gamma[3:6, 6:], 0.0)
    assert np.allclose(Gamma[6:9, :6], 0.0)
    assert np.allclose(Gamma[6:9, 9:], 0.0)
    assert np.allclose(Gamma[9:12, :9], 0.0)
    assert np.allclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ R.T, np.eye(3))
    beam_dir = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    assert np.allclose(R[:, 0], beam_dir)
    ref_vec = np.array([1.0, 0.0, 0.0])
    Gamma_custom = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ref_vec)
    R_custom = Gamma_custom[:3, :3]
    expected_local_y = np.array([0.0, 0.0, -1.0])
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
    non_unit_vector = np.array([2.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, non_unit_vector)
    parallel_vector = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, parallel_vector)
    with pytest.raises(ValueError):
        fcn(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, None)
    wrong_shape_vector = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, wrong_shape_vector)