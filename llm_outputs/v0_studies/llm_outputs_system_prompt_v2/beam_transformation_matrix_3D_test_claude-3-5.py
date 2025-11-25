def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes."""
    tol = 1e-14
    gamma = fcn(0, 0, 0, 1, 0, 0, np.array([0, 0, 1]))
    R = gamma[0:3, 0:3]
    assert np.allclose(R[:, 0], [1, 0, 0], atol=tol)
    assert np.allclose(R[:, 1], [0, 1, 0], atol=tol)
    assert np.allclose(R[:, 2], [0, 0, 1], atol=tol)
    gamma = fcn(0, 0, 0, 0, 1, 0, np.array([0, 0, 1]))
    R = gamma[0:3, 0:3]
    assert np.allclose(R[:, 0], [0, 1, 0], atol=tol)
    assert np.allclose(R[:, 1], [-1, 0, 0], atol=tol)
    assert np.allclose(R[:, 2], [0, 0, 1], atol=tol)
    gamma = fcn(0, 0, 0, 0, 0, 1, np.array([0, 1, 0]))
    R = gamma[0:3, 0:3]
    assert np.allclose(R[:, 0], [0, 0, 1], atol=tol)
    assert np.allclose(R[:, 1], [0, 1, 0], atol=tol)
    assert np.allclose(R[:, 2], [-1, 0, 0], atol=tol)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices."""
    tol = 1e-14
    gamma = fcn(0, 0, 0, 1, 1, 0, np.array([0, 0, 1]))
    R = gamma[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=tol)
    assert np.allclose(np.linalg.det(R), 1.0, atol=tol)
    assert np.allclose(gamma[0:3, 0:3], gamma[3:6, 3:6], atol=tol)
    assert np.allclose(gamma[0:3, 0:3], gamma[6:9, 6:9], atol=tol)
    assert np.allclose(gamma[0:3, 0:3], gamma[9:12, 9:12], atol=tol)
    gamma = fcn(0, 0, 0, 1, 2, 3, np.array([0, 0, 1]))
    R = gamma[0:3, 0:3]
    assert np.allclose(R @ R.T, np.eye(3), atol=tol)
    assert np.allclose(np.linalg.det(R), 1.0, atol=tol)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions."""
    with pytest.raises(ValueError, match='reference vector must be a unit vector'):
        fcn(0, 0, 0, 1, 0, 0, np.array([0, 0, 2]))
    with pytest.raises(ValueError, match='reference vector cannot be parallel'):
        fcn(0, 0, 0, 1, 0, 0, np.array([1, 0, 0]))
    with pytest.raises(ValueError):
        fcn(0, 0, 0, 1, 0, 0, np.array([0, 0, 1, 0]))
    with pytest.raises(ValueError, match='beam length cannot be zero'):
        fcn(1, 1, 1, 1, 1, 1, np.array([0, 0, 1]))