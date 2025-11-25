def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    Gamma_x = fcn(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_x = Gamma_x[0:3, 0:3]
    expected_R_x = np.eye(3)
    assert np.allclose(R_x, expected_R_x, atol=1e-12)
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 5.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_y = Gamma_y[0:3, 0:3]
    expected_R_y = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R_y, expected_R_y, atol=1e-12)
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 3.0, None)
    R_z = Gamma_z[0:3, 0:3]
    expected_R_z = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(R_z, expected_R_z, atol=1e-12)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    (x1, y1, z1) = (1.0, 2.0, 3.0)
    (x2, y2, z2) = (4.0, 6.0, 3.0)
    Gamma = fcn(x1, y1, z1, x2, y2, z2, None)
    dx = np.array([x2 - x1, y2 - y1, z2 - z1], dtype=float)
    x_local = dx / np.linalg.norm(dx)
    ref = np.array([0.0, 0.0, 1.0])
    y_local = np.cross(ref, x_local)
    y_local = y_local / np.linalg.norm(y_local)
    z_local = np.cross(x_local, y_local)
    L = np.column_stack((x_local, y_local, z_local))
    expected_R = L.T
    R = Gamma[0:3, 0:3]
    assert np.allclose(R, expected_R, atol=1e-12)
    I3 = np.eye(3)
    assert np.allclose(R @ R.T, I3, atol=1e-12)
    assert np.allclose(R.T @ R, I3, atol=1e-12)
    detR = np.linalg.det(R)
    assert np.isclose(detR, 1.0, atol=1e-12)
    assert Gamma.shape == (12, 12)
    for i in range(4):
        block = Gamma[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)]
        assert np.allclose(block, R, atol=1e-12)
    zero = np.zeros((3, 3))
    for i in range(4):
        for j in range(4):
            if i != j:
                block = Gamma[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
                assert np.allclose(block, zero, atol=1e-12)
    I12 = np.eye(12)
    assert np.allclose(Gamma @ Gamma.T, I12, atol=1e-12)
    assert np.allclose(Gamma.T @ Gamma, I12, atol=1e-12)
    u_l = np.arange(1.0, 13.0)
    u_g_expected = np.zeros(12)
    for i in range(4):
        u_g_expected[3 * i:3 * (i + 1)] = L @ u_l[3 * i:3 * (i + 1)]
    u_g = Gamma.T @ u_l
    assert np.allclose(u_g, u_g_expected, atol=1e-12)

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
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None)