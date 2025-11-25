def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes."""
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, None)
    expected_x = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    np.testing.assert_array_almost_equal(Gamma_x[:3, :3], expected_x)
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, None)
    expected_y = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_almost_equal(Gamma_y[:3, :3], expected_y)
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, None)
    expected_z = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_almost_equal(Gamma_z[:3, :3], expected_z)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices."""
    ref_vec = np.array([0, 1, 0])
    Gamma = fcn(0, 0, 0, 1, 0, 0, ref_vec)
    assert Gamma.shape == (12, 12)
    R = Gamma[:3, :3]
    for i in range(1, 4):
        block = Gamma[3 * i:3 * i + 3, 3 * i:3 * i + 3]
        np.testing.assert_array_almost_equal(block, R)
    np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
    np.testing.assert_array_almost_equal(np.linalg.det(R), 1.0)
    Gamma_diag = fcn(0, 0, 0, 1, 1, 1, np.array([0, 0, 1]))
    R_diag = Gamma_diag[:3, :3]
    expected_x = np.array([1, 1, 1]) / np.sqrt(3)
    np.testing.assert_array_almost_equal(R_diag[0, :], expected_x)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions."""
    with pytest.raises(ValueError, match='unit vector'):
        fcn(0, 0, 0, 1, 0, 0, np.array([2, 0, 0]))
    with pytest.raises(ValueError, match='parallel'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError, match='shape'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0]))
    with pytest.raises(ValueError, match='zero length'):
        fcn(0, 0, 0, 0, 0, 0, None)