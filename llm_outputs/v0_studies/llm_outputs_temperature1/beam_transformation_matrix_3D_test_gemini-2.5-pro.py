def test_cardinal_axis_alignment(fcn: Callable):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    gamma_x = fcn(0, 0, 0, 10, 0, 0, reference_vector=None)
    R_x_expected = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    assert np.allclose(gamma_x[0:3, 0:3], R_x_expected)
    gamma_y = fcn(0, 0, 0, 0, 10, 0, reference_vector=None)
    R_y_expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(gamma_y[0:3, 0:3], R_y_expected)
    gamma_z = fcn(0, 0, 0, 0, 0, 10, reference_vector=None)
    R_z_expected = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert np.allclose(gamma_z[0:3, 0:3], R_z_expected)

def test_transformation_matrix_properties(fcn: Callable):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    (x1, y1, z1) = (1, 2, 3)
    (x2, y2, z2) = (5, 7, 11)
    ref_vec = np.array([0.1, 0.2, 0.3])
    ref_vec /= np.linalg.norm(ref_vec)
    gamma = fcn(x1, y1, z1, x2, y2, z2, reference_vector=ref_vec)
    assert np.allclose(gamma.T @ gamma, np.eye(12))
    R = gamma[0:3, 0:3]
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R, gamma[3:6, 3:6])
    assert np.allclose(R, gamma[6:9, 6:9])
    assert np.allclose(R, gamma[9:12, 9:12])
    assert np.allclose(gamma[0:3, 3:6], np.zeros((3, 3)))
    assert np.allclose(gamma[6:9, 0:3], np.zeros((3, 3)))

def test_beam_transformation_matrix_error_messages(fcn: Callable):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for two invalid reference vector conditions:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    Also veriefies proper error handling for:
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError
    """
    with pytest.raises(ValueError, match='unit vector'):
        ref_vec_non_unit = np.array([1.0, 1.0, 0.0])
        fcn(0, 0, 0, 1, 0, 0, reference_vector=ref_vec_non_unit)
    with pytest.raises(ValueError, match='parallel to the beam axis'):
        beam_axis = np.array([1.0, 1.0, 1.0])
        ref_vec_parallel = beam_axis / np.linalg.norm(beam_axis)
        fcn(0, 0, 0, 5, 5, 5, reference_vector=ref_vec_parallel)
    with pytest.raises(ValueError, match='non-zero length'):
        fcn(1, 2, 3, 1, 2, 3, reference_vector=None)
    with pytest.raises(ValueError, match='shape \\(3,\\)'):
        ref_vec_bad_shape = np.array([1.0, 0.0])
        fcn(0, 0, 0, 1, 0, 0, reference_vector=ref_vec_bad_shape)