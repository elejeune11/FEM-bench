def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 5, 0, 0, reference_vector=None)
    lambda_x = gamma_x[0:3, 0:3]
    expected_lambda_x = np.identity(3)
    assert np.allclose(lambda_x, expected_lambda_x)
    gamma_y = fcn(0, 0, 0, 0, 5, 0, reference_vector=None)
    lambda_y = gamma_y[0:3, 0:3]
    expected_lambda_y = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(lambda_y, expected_lambda_y)
    gamma_z = fcn(0, 0, 0, 0, 0, 5, reference_vector=None)
    lambda_z = gamma_z[0:3, 0:3]
    expected_lambda_z = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert np.allclose(lambda_z, expected_lambda_z)
    gamma_nx = fcn(1, 1, 1, -4, 1, 1, reference_vector=None)
    lambda_nx = gamma_nx[0:3, 0:3]
    expected_lambda_nx = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert np.allclose(lambda_nx, expected_lambda_nx)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma1 = fcn(1, 1, 0, 2, 2, 0, reference_vector=None)
    lambda1 = gamma1[0:3, 0:3]
    for i in range(4):
        for j in range(4):
            block = gamma1[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
            if i == j:
                assert np.allclose(block, lambda1)
            else:
                assert np.allclose(block, np.zeros((3, 3)))
    assert np.allclose(lambda1.T @ lambda1, np.identity(3))
    assert np.allclose(gamma1.T @ gamma1, np.identity(12))
    assert np.isclose(np.linalg.det(lambda1), 1.0)
    s2 = np.sqrt(2)
    expected_lambda1 = np.array([[1 / s2, -1 / s2, 0], [1 / s2, 1 / s2, 0], [0, 0, 1]])
    assert np.allclose(lambda1, expected_lambda1)
    ref_vec = np.array([0, 1, 0], dtype=float)
    gamma2 = fcn(0, 0, 0, 1, 1, 1, reference_vector=ref_vec)
    lambda2 = gamma2[0:3, 0:3]
    for i in range(4):
        for j in range(4):
            block = gamma2[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
            if i == j:
                assert np.allclose(block, lambda2)
            else:
                assert np.allclose(block, np.zeros((3, 3)))
    assert np.allclose(lambda2.T @ lambda2, np.identity(3))
    assert np.allclose(gamma2.T @ gamma2, np.identity(12))
    assert np.isclose(np.linalg.det(lambda2), 1.0)
    s2 = np.sqrt(2)
    s3 = np.sqrt(3)
    s6 = np.sqrt(6)
    expected_lambda2 = np.array([[1 / s3, 1 / s2, -1 / s6], [1 / s3, 0, 2 / s6], [1 / s3, -1 / s2, -1 / s6]])
    assert np.allclose(lambda2, expected_lambda2)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    with pytest.raises(ValueError, match='reference_vector must be a unit vector'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([0, 2.0, 0]))
    with pytest.raises(ValueError, match='reference_vector is parallel to the beam axis'):
        ref_vec_parallel = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        fcn(0, 0, 0, 1, 1, 0, reference_vector=ref_vec_parallel)
    with pytest.raises(ValueError, match='reference_vector is parallel to the beam axis'):
        ref_vec_antiparallel = -np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        fcn(0, 0, 0, 1, 1, 0, reference_vector=ref_vec_antiparallel)
    with pytest.raises(ValueError, match='Beam has zero length'):
        fcn(1, 2, 3, 1, 2, 3, reference_vector=None)
    with pytest.raises(ValueError, match="reference_vector doesn't have shape \\(3,\\)"):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1, 0]))
    with pytest.raises(ValueError, match="reference_vector doesn't have shape \\(3,\\)"):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([1, 0, 0, 0]))
    with pytest.raises(ValueError, match="reference_vector doesn't have shape \\(3,\\)"):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([[1], [0], [0]]))