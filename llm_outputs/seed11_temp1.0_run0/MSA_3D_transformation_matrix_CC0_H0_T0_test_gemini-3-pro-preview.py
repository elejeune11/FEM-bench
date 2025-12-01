def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, None)
    expected_R_x = np.eye(3)
    for i in range(4):
        block = gamma_x[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)]
        np.testing.assert_allclose(block, expected_R_x, atol=1e-10, err_msg=f'X-axis beam: Diagonal block {i} mismatch')
    gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, None)
    expected_R_y = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for i in range(4):
        block = gamma_y[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)]
        np.testing.assert_allclose(block, expected_R_y, atol=1e-10, err_msg=f'Y-axis beam: Diagonal block {i} mismatch')
    gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None)
    expected_R_z = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for i in range(4):
        block = gamma_z[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)]
        np.testing.assert_allclose(block, expected_R_z, atol=1e-10, err_msg=f'Z-axis beam: Diagonal block {i} mismatch')

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    (x1, y1, z1) = (1.0, 2.0, 3.0)
    (x2, y2, z2) = (4.0, 6.0, 8.0)
    gamma = fcn(x1, y1, z1, x2, y2, z2, None)
    assert gamma.shape == (12, 12), 'Transformation matrix should be 12x12'
    R = gamma[0:3, 0:3]
    identity = np.eye(3)
    np.testing.assert_allclose(R.T @ R, identity, atol=1e-10, err_msg='Rotation matrix R is not orthogonal')
    np.testing.assert_allclose(R @ R.T, identity, atol=1e-10, err_msg='Rotation matrix R is not orthogonal')
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 1e-10, f'Determinant of rotation matrix should be 1.0, got {det}'
    zeros = np.zeros((3, 3))
    for i in range(4):
        for j in range(4):
            block = gamma[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
            if i == j:
                np.testing.assert_allclose(block, R, atol=1e-10, err_msg=f'Diagonal block ({i},{j}) mismatch')
            else:
                np.testing.assert_allclose(block, zeros, atol=1e-10, err_msg=f'Off-diagonal block ({i},{j}) should be zero')
    ref_vec = np.array([0.0, 1.0, 0.0])
    gamma_spec = fcn(0.0, 0.0, 0.0, 10.0, 0.0, 0.0, ref_vec)
    R_spec = gamma_spec[0:3, 0:3]
    expected_R_spec = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(R_spec, expected_R_spec, atol=1e-10, err_msg='Explicit reference vector example mismatch')

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    with pytest.raises(ValueError):
        non_unit_ref = np.array([1.0, 1.0, 0.0])
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, non_unit_ref)
    with pytest.raises(ValueError):
        parallel_ref = np.array([1.0, 0.0, 0.0])
        fcn(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, parallel_ref)
    with pytest.raises(ValueError):
        fcn(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None)
    with pytest.raises(ValueError):
        bad_shape_ref = np.array([0.0, 1.0])
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, bad_shape_ref)