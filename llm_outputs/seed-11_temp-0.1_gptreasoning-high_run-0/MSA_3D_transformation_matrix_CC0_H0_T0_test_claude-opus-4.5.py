def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    R_x = Gamma_x[0:3, 0:3]
    expected_R_x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(R_x, expected_R_x, atol=1e-10), f'X-axis beam failed: {R_x}'
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    R_y = Gamma_y[0:3, 0:3]
    expected_R_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(R_y, expected_R_y, atol=1e-10), f'Y-axis beam failed: {R_y}'
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    R_z = Gamma_z[0:3, 0:3]
    expected_R_z = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert np.allclose(R_z, expected_R_z, atol=1e-10), f'Z-axis beam failed: {R_z}'

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    Gamma1 = fcn(0, 0, 0, 5, 0, 0, None)
    assert Gamma1.shape == (12, 12), f'Expected shape (12, 12), got {Gamma1.shape}'
    identity_check1 = Gamma1.T @ Gamma1
    assert np.allclose(identity_check1, np.eye(12), atol=1e-10), 'Transformation matrix is not orthogonal'
    identity_check2 = Gamma1 @ Gamma1.T
    assert np.allclose(identity_check2, np.eye(12), atol=1e-10), 'Transformation matrix is not orthogonal'
    R1 = Gamma1[0:3, 0:3]
    det_R1 = np.linalg.det(R1)
    assert np.isclose(det_R1, 1.0, atol=1e-10), f'Determinant of rotation matrix should be 1, got {det_R1}'
    Gamma2 = fcn(0, 0, 0, 1, 1, 1, None)
    assert Gamma2.shape == (12, 12)
    assert np.allclose(Gamma2.T @ Gamma2, np.eye(12), atol=1e-10)
    R2 = Gamma2[0:3, 0:3]
    det_R2 = np.linalg.det(R2)
    assert np.isclose(det_R2, 1.0, atol=1e-10)
    Gamma3 = fcn(1, 2, 3, 4, 5, 6, None)
    R = Gamma3[0:3, 0:3]
    assert np.allclose(Gamma3[0:3, 0:3], R, atol=1e-10)
    assert np.allclose(Gamma3[3:6, 3:6], R, atol=1e-10)
    assert np.allclose(Gamma3[6:9, 6:9], R, atol=1e-10)
    assert np.allclose(Gamma3[9:12, 9:12], R, atol=1e-10)
    assert np.allclose(Gamma3[0:3, 3:6], np.zeros((3, 3)), atol=1e-10)
    assert np.allclose(Gamma3[0:3, 6:9], np.zeros((3, 3)), atol=1e-10)
    assert np.allclose(Gamma3[0:3, 9:12], np.zeros((3, 3)), atol=1e-10)
    ref_vec = np.array([0, 1, 0])
    Gamma4 = fcn(0, 0, 0, 1, 0, 0, ref_vec)
    assert Gamma4.shape == (12, 12)
    assert np.allclose(Gamma4.T @ Gamma4, np.eye(12), atol=1e-10)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    non_unit_vector = np.array([1, 1, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, non_unit_vector)
    non_unit_vector2 = np.array([0.5, 0, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 1, 0, non_unit_vector2)
    parallel_vector = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, parallel_vector)
    parallel_vector_y = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 5, 0, parallel_vector_y)
    anti_parallel_vector = np.array([-1, 0, 0])
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, anti_parallel_vector)
    with pytest.raises(ValueError):
        fcn(1, 2, 3, 1, 2, 3, None)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 0, None)