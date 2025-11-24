def test_cardinal_axis_alignment(fcn: Callable[[float, float, float, float, float, float, Optional[np.ndarray]], np.ndarray]):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 5, 0, 0, reference_vector=None)
    R_x_actual = gamma_x[0:3, 0:3]
    R_x_expected = np.identity(3)
    np.testing.assert_allclose(R_x_actual, R_x_expected, atol=1e-15)
    gamma_y = fcn(1, 1, 1, 1, 6, 1, reference_vector=None)
    R_y_actual = gamma_y[0:3, 0:3]
    R_y_expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    np.testing.assert_allclose(R_y_actual, R_y_expected, atol=1e-15)
    gamma_z = fcn(2, 3, 4, 2, 3, 10, reference_vector=None)
    R_z_actual = gamma_z[0:3, 0:3]
    R_z_expected = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(R_z_actual, R_z_expected, atol=1e-15)

def test_transformation_matrix_properties(fcn: Callable[[float, float, float, float, float, float, Optional[np.ndarray]], np.ndarray]):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    gamma1 = fcn(1, 1, 1, 2, 3, 4, reference_vector=None)
    R1 = gamma1[0:3, 0:3]
    np.testing.assert_allclose(R1.T @ R1, np.identity(3), atol=1e-15)
    assert np.isclose(np.linalg.det(R1), 1.0)
    expected_gamma1 = np.kron(np.identity(4, dtype=int), R1)
    np.testing.assert_allclose(gamma1, expected_gamma1, atol=1e-15)
    ref_vec = np.array([0.0, 1.0, 0.0])
    gamma2 = fcn(0, 0, 0, 1, 1, 0, reference_vector=ref_vec)
    R2 = gamma2[0:3, 0:3]
    lx = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    ly = np.array([0.0, 0.0, -1.0])
    lz = np.cross(lx, ly)
    R2_expected = np.column_stack([lx, ly, lz])
    np.testing.assert_allclose(R2, R2_expected, atol=1e-15)
    np.testing.assert_allclose(R2.T @ R2, np.identity(3), atol=1e-15)
    assert np.isclose(np.linalg.det(R2), 1.0)

def test_beam_transformation_matrix_error_messages(fcn: Callable[[float, float, float, float, float, float, Optional[np.ndarray]], np.ndarray]):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    with pytest.raises(ValueError, match='must be a unit vector'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match='parallel to the beam axis'):
        fcn(0, 0, 0, 1, 1, 1, reference_vector=np.array([1.0, 1.0, 1.0]) / np.sqrt(3))
    with pytest.raises(ValueError, match='must have shape \\(3,\\)'):
        fcn(0, 0, 0, 1, 0, 0, reference_vector=np.array([0.0, 0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match='zero length'):
        fcn(5, 4, 3, 5, 4, 3, reference_vector=None)