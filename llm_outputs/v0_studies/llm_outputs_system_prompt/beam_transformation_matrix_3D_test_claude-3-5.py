def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes."""
    import numpy as np
    gamma_x = fcn(0, 0, 0, 1, 0, 0, np.array([0, 0, 1]))
    assert np.allclose(gamma_x[0:3, 0:3], np.eye(3))
    gamma_y = fcn(0, 0, 0, 0, 1, 0, np.array([0, 0, 1]))
    expected_y = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(gamma_y[0:3, 0:3], expected_y)
    gamma_z = fcn(0, 0, 0, 0, 0, 1, np.array([0, 1, 0]))
    expected_z = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.allclose(gamma_z[0:3, 0:3], expected_z)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices."""
    import numpy as np
    gamma_45 = fcn(0, 0, 0, 1, 1, 0, np.array([0, 0, 1]))
    assert gamma_45.shape == (12, 12)
    assert np.allclose(gamma_45[0:3, 0:3], gamma_45[3:6, 3:6])
    assert np.allclose(gamma_45[0:3, 0:3], gamma_45[6:9, 6:9])
    assert np.allclose(gamma_45[0:3, 0:3], gamma_45[9:12, 9:12])
    R = gamma_45[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(np.linalg.det(R), 1.0)
    gamma = fcn(1, 2, 3, 4, 6, 8, np.array([0, 0, 1]))
    R = gamma[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(np.linalg.det(R), 1.0)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions."""
    import numpy as np
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([0, 0, 2]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([0, 0, 1, 0]))
    with pytest.raises(ValueError):
        fcn(1, 1, 1, 1, 1, 1, np.array([0, 0, 1]))