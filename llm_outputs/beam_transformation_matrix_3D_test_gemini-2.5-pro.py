def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 5, 0, 0, reference_vector=None)
    lambda_x = gamma_x[:3, :3]
    expected_lambda_x = np.identity(3)
    assert np.allclose(lambda_x, expected_lambda_x)
    gamma_y = fcn(0, 0, 0, 0, 5, 0, reference_vector=None)
    lambda_y = gamma_y[:3, :3]
    expected_lambda_y = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    assert np.allclose(lambda_y, expected_lambda_y)
    gamma_z = fcn(0, 0, 0, 0, 0, 5, reference_vector=None)
    lambda_z = gamma_z[:3, :3]
    expected_lambda_z = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert np.allclose(lambda_z, expected_lambda_z)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma1 = fcn(1, 2, 3, 4, 6, 8, reference_vector=None)
    assert gamma1.shape == (12, 12)
    identity_12 = np.identity(12)
    assert np.allclose(gamma1.T @ gamma1, identity_12)
    lambda1 = gamma1[:3, :3]
    expected_gamma1 = np.kron(np.identity(4), lambda1)
    assert np.allclose(gamma1, expected_gamma1)
    ref_vec = np.array([0.0, 1.0, 0.0])
    gamma2 = fcn(0, 0, 0, 1, 1, 1, reference_vector=ref_vec)
    assert gamma2.shape == (12, 12)
    assert np.allclose(gamma2.T @ gamma2, identity_12)
    lambda2 = gamma2[:3, :3]
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    sqrt6 = np.sqrt(6)
    expected_lambda2 = np.array([[1 / sqrt3, 1 / sqrt2, -1 / sqrt6], [1 / sqrt3, 0, 2 / sqrt6], [1 / sqrt3, -1 / sqrt2, -1 / sqrt6]])
    assert np.allclose(lambda2, expected_lambda2)

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also verifies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    with pytest.raises(ValueError, match='unit vector'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match='parallel to the beam axis'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match='parallel to the beam axis'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([-1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match='zero length'):
        fcn(1, 2, 3, 1, 2, 3, reference_vector=None)
    with pytest.raises(ValueError, match='shape \\(3,\\)'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match='shape \\(3,\\)'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([[1.0], [0.0], [0.0]]))