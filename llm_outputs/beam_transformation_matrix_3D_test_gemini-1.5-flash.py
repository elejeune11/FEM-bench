def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes."""
    Gamma_x = fcn(0, 0, 0, 1, 0, 0, np.array([0, 0, 1]))
    assert np.allclose(Gamma_x[:3, :3], np.eye(3))
    Gamma_y = fcn(0, 0, 0, 0, 1, 0, np.array([0, 0, 1]))
    assert np.allclose(Gamma_y[:3, :3], np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    Gamma_z = fcn(0, 0, 0, 0, 0, 1, np.array([0, 1, 0]))
    assert np.allclose(Gamma_z[:3, :3], np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations."""
    Gamma1 = fcn(0, 0, 0, 1, 1, 1, np.array([0, 0, 1]))
    assert np.allclose(Gamma1 @ Gamma1.T, np.eye(12))
    Gamma2 = fcn(1, 2, 3, 4, 5, 6, np.array([1, 0, 0]))
    assert np.allclose(Gamma2.T @ Gamma2, np.eye(12))

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid reference vectors."""
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 1, 1]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 0, 0, 0, np.array([0, 0, 1]))