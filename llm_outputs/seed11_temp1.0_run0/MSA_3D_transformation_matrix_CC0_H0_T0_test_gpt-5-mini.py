def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    """
    Gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_x_expected = np.eye(3)
    assert _block_diag_equal(Gamma_x, R_x_expected)
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_y_expected = np.column_stack(([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    assert _block_diag_equal(Gamma_y, R_y_expected)
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.array([0.0, 1.0, 0.0]))
    R_z_expected = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    assert _block_diag_equal(Gamma_z, R_z_expected)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Verifies orthonormality of the 3x3 direction cosine submatrix, right-handedness (determinant = +1),
    block-diagonal repetition, and correct mapping of the local axial vector to the global beam axis.
    """
    beams = [((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), np.array([0.0, 0.0, 1.0])), ((-1.0, 0.5, 2.0), (2.0, -1.0, 5.0), None), ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), np.array([0.0, 0.0, 1.0])), ((0.0, 1.0, 0.0), (1.0, 2.0, 3.0), _normalized(np.array([0.1, 0.8, 0.6])))]
    for ((x1, y1, z1), (x2, y2, z2), ref) in beams:
        Gamma = fcn(x1, y1, z1, x2, y2, z2, ref)
        assert Gamma.shape == (12, 12)
        R = Gamma[:3, :3]
        assert _is_rotation_matrix(R)
        assert _block_diag_equal(Gamma, R)
        beam_axis = _normalized(np.array([x2 - x1, y2 - y1, z2 - z1], dtype=float))
        mapped_axis = R @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(mapped_axis, beam_axis, atol=1e-12)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises ValueError for invalid reference vectors and zero-length beams.
    Cases:
    1. Non-unit reference vector magnitude
    2. Reference vector parallel to beam axis
    3. Zero-length beam (start and end coincide)
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, np.array([0.0, 0.0, 1.0]))