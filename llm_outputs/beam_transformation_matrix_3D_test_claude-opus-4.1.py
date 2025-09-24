def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma = fcn(0, 0, 0, 1, 0, 0, None)
    expected_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.allclose(gamma[:3, :3], expected_rotation)
    assert np.allclose(gamma[3:6, 3:6], expected_rotation)
    assert np.allclose(gamma[6:9, 6:9], expected_rotation)
    assert np.allclose(gamma[9:12, 9:12], expected_rotation)
    gamma = fcn(0, 0, 0, 0, 1, 0, None)
    expected_rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert np.allclose(gamma[:3, :3], expected_rotation)
    assert np.allclose(gamma[3:6, 3:6], expected_rotation)
    assert np.allclose(gamma[6:9, 6:9], expected_rotation)
    assert np.allclose(gamma[9:12, 9:12], expected_rotation)
    gamma = fcn(0, 0, 0, 0, 0, 1, None)
    expected_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(gamma[:3, :3], expected_rotation)
    assert np.allclose(gamma[3:6, 3:6], expected_rotation)
    assert np.allclose(gamma[6:9, 6:9], expected_rotation)
    assert np.allclose(gamma[9:12, 9:12], expected_rotation)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma1 = fcn(0, 0, 0, 1, 1, 1, None)
    assert np.allclose(gamma1.T @ gamma1, np.eye(12))
    gamma2 = fcn(-1, 2, 3, 4, -5, 6, None)
    assert np.allclose(gamma2.T @ gamma2, np.eye(12))
    ref_vec = np.array([0, 1, 0]) / np.linalg.norm([0, 1, 0])
    gamma3 = fcn(0, 0, 0, 1, 0, 0, ref_vec)
    assert np.allclose(gamma3.T @ gamma3, np.eye(12))
    ref_vec = np.array([1, 0, 0]) / np.linalg.norm([1, 0, 0])
    gamma4 = fcn(0, 0, 0, 1, 1, 1, ref_vec)
    assert np.allclose(gamma4.T @ gamma4, np.eye(12))
    gamma5 = fcn(1, 2, 3, 4, 5, 6, None)
    rotation_block = gamma5[:3, :3]
    assert np.allclose(gamma5[3:6, 3:6], rotation_block)
    assert np.allclose(gamma5[6:9, 6:9], rotation_block)
    assert np.allclose(gamma5[9:12, 9:12], rotation_block)
    assert np.allclose(gamma5[:3, 3:], np.zeros((3, 9)))
    assert np.allclose(gamma5[3:6, :3], np.zeros((3, 3)))
    assert np.allclose(gamma5[3:6, 6:], np.zeros((3, 6)))
    assert np.allclose(gamma5[6:9, :6], np.zeros((3, 6)))
    assert np.allclose(gamma5[6:9, 9:], np.zeros((3, 3)))
    assert np.allclose(gamma5[9:12, :9], np.zeros((3, 9)))

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    non_unit_vec = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, non_unit_vec)
    beam_direction = np.array([1, 0, 0])
    parallel_ref_vec = beam_direction / np.linalg.norm(beam_direction)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, parallel_ref_vec)
    beam_direction2 = np.array([1, 1, 1])
    parallel_ref_vec2 = beam_direction2 / np.linalg.norm(beam_direction2)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 1, 1, parallel_ref_vec2)
    with pytest.raises(ValueError):
        fcn(1, 2, 3, 1, 2, 3, None)
    ref_vec = np.array([0, 0, 1]) / np.linalg.norm([0, 0, 1])
    with pytest.raises(ValueError):
        fcn(5, 5, 5, 5, 5, 5, ref_vec)