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
    gamma4 = fcn(1, 2, 3, 4, 5, 6, None)
    rotation = gamma4[:3, :3]
    assert np.allclose(gamma4[3:6, 3:6], rotation)
    assert np.allclose(gamma4[6:9, 6:9], rotation)
    assert np.allclose(gamma4[9:12, 9:12], rotation)
    for i in range(4):
        for j in range(4):
            if i != j:
                assert np.allclose(gamma4[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3], np.zeros((3, 3)))
    gamma5 = fcn(0, 0, 0, 3, 4, 0, None)
    rotation = gamma5[:3, :3]
    assert np.allclose(np.linalg.det(rotation), 1.0)

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
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 1, 1]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 1, 0, np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 1, np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        fcn(1, 2, 3, 1, 2, 3, None)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 0, None)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([[1], [0], [0]]))