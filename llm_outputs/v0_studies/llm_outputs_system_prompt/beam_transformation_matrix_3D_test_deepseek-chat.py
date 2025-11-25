def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x = np.eye(3)
    assert np.allclose(Gamma_x[:3, :3], expected_x)
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(Gamma_y[:3, :3], expected_y)
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(Gamma_z[:3, :3], expected_z)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    ref_vec = np.array([0, 1, 0])
    Gamma = fcn(0, 0, 0, 1, 0, 0, ref_vec)
    Gamma_3x3 = Gamma[:3, :3]
    assert np.allclose(Gamma_3x3.T @ Gamma_3x3, np.eye(3))
    assert np.allclose(Gamma.T @ Gamma, np.eye(12))
    Gamma_diag = fcn(0, 0, 0, 1, 1, 1, None)
    Gamma_diag_3x3 = Gamma_diag[:3, :3]
    assert np.allclose(Gamma_diag_3x3.T @ Gamma_diag_3x3, np.eye(3))
    Gamma_known = fcn(0, 0, 0, 2, 0, 0, np.array([0, 0, 1]))
    expected_identity = np.eye(3)
    assert np.allclose(Gamma_known[:3, :3], expected_identity)

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
        fcn(0, 0, 0, 1, 0, 0, np.array([2, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 0, None)
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))