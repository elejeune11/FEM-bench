def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 1, 0, 0, reference_vector=None)
    lambda_x = gamma_x[0:3, 0:3]
    expected_lambda_x = np.eye(3)
    assert np.allclose(lambda_x, expected_lambda_x)
    gamma_y = fcn(0, 0, 0, 0, 1, 0, reference_vector=None)
    lambda_y = gamma_y[0:3, 0:3]
    expected_lambda_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])
    assert np.allclose(lambda_y, expected_lambda_y)
    gamma_z = fcn(0, 0, 0, 0, 0, 1, reference_vector=None)
    lambda_z = gamma_z[0:3, 0:3]
    expected_lambda_z = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(lambda_z, expected_lambda_z)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma = fcn(1, 2, 3, 4, 6, 8, reference_vector=None)
    lambda_matrix = gamma[0:3, 0:3]
    assert np.allclose(lambda_matrix.T @ lambda_matrix, np.eye(3))
    assert np.isclose(np.linalg.det(lambda_matrix), 1.0)
    expected_gamma = np.zeros((12, 12))
    for i in range(4):
        expected_gamma[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = lambda_matrix
    assert np.allclose(gamma, expected_gamma)
    gamma_45 = fcn(0, 0, 0, 1, 1, 0, reference_vector=None)
    lambda_45 = gamma_45[0:3, 0:3]
    c = 1 / np.sqrt(2)
    expected_lambda_45 = np.array([[c, c, 0], [-c, c, 0], [0, 0, 1]])
    assert np.allclose(lambda_45, expected_lambda_45)

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
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([0, 2.0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1.0, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 5, 0, reference_vector=np.array([0, -1.0, 0]))
    with pytest.raises(ValueError):
        fcn(1, 2, 3, 1, 2, 3, reference_vector=None)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1.0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1.0, 0, 0, 0]))