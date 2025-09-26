def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 10, 0, 0, reference_vector=None)
    R_x = gamma_x[:3, :3]
    R_x_expected = np.identity(3)
    assert np.allclose(R_x, R_x_expected)
    gamma_y = fcn(0, 0, 0, 0, 5, 0, reference_vector=None)
    R_y = gamma_y[:3, :3]
    R_y_expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])
    assert np.allclose(R_y, R_y_expected)
    gamma_z = fcn(0, 0, 0, 0, 0, 2, reference_vector=None)
    R_z = gamma_z[:3, :3]
    R_z_expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert np.allclose(R_z, R_z_expected)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    s2 = np.sqrt(2.0)
    gamma_45 = fcn(0, 0, 0, 1, 1, 0, reference_vector=None)
    R_45 = gamma_45[:3, :3]
    R_45_expected = np.array([[1 / s2, -1 / s2, 0], [1 / s2, 1 / s2, 0], [0, 0, 1]])
    assert np.allclose(R_45, R_45_expected)
    gamma_gen = fcn(1, 2, 3, 4, 6, 15, reference_vector=None)
    local_x = np.array([3, 4, 12]) / 13.0
    ref_vec = np.array([0, 0, 1])
    local_y = np.cross(ref_vec, local_x)
    local_y /= np.linalg.norm(local_y)
    local_z = np.cross(local_x, local_y)
    R_gen_expected = np.vstack((local_x, local_y, local_z)).T
    assert np.allclose(gamma_gen[:3, :3], R_gen_expected)
    test_cases = [((0, 0, 0, 1, 1, 1), None), ((5, -1, 2, -3, 4, 8), None), ((1, 1, 0, 2, 2, 0), np.array([0, 0, 1.0])), ((0, 0, 1, 0, 0, 5), np.array([1.0, 0, 0]))]
    for (coords, ref_vec) in test_cases:
        gamma = fcn(*coords, reference_vector=ref_vec)
        assert gamma.shape == (12, 12)
        assert np.allclose(gamma.T @ gamma, np.identity(12))
        R = gamma[:3, :3]
        assert np.isclose(np.linalg.det(R), 1.0)
        expected_gamma = np.kron(np.eye(4), R)
        assert np.allclose(gamma, expected_gamma)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    with pytest.raises(ValueError, match='Beam has zero length'):
        fcn(1, 2, 3, 1, 2, 3, reference_vector=None)
    with pytest.raises(ValueError, match='reference_vector must be a unit vector'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1, 1, 0]))
    with pytest.raises(ValueError, match='reference_vector must be a unit vector'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match='reference_vector cannot be parallel to the beam axis'):
        fcn(0, 0, 0, 5, 0, 0, reference_vector=np.array([1, 0, 0]))
    with pytest.raises(ValueError, match='reference_vector cannot be parallel to the beam axis'):
        fcn(1, 1, 1, 3, 3, 3, reference_vector=np.array([-1, -1, -1]) / np.sqrt(3))
    with pytest.raises(ValueError, match='reference_vector must have shape \\(3,\\)'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1, 0]))
    with pytest.raises(ValueError, match='reference_vector must have shape \\(3,\\)'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1, 0, 0, 0]))
    with pytest.raises(ValueError, match='reference_vector must have shape \\(3,\\)'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([[1], [0], [0]]))