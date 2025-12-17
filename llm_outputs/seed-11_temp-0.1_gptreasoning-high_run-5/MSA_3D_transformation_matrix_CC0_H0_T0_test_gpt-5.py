def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    The expected orientations are determined by:
    Test cases:
    """
    z_hat = np.array([0.0, 0.0, 1.0])
    y_hat = np.array([0.0, 1.0, 0.0])

    def expected_R(p1, p2, ref):
        v = np.array(p2) - np.array(p1)
        lx = v / np.linalg.norm(v)
        ly = np.cross(ref, lx)
        ly = ly / np.linalg.norm(ly)
        lz = np.cross(lx, ly)
        return np.column_stack((lx, ly, lz))
    Gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, z_hat)
    R_x = Gamma_x[:3, :3]
    E_x = expected_R([0, 0, 0], [1, 0, 0], z_hat)
    assert R_x.shape == (3, 3)
    assert np.allclose(R_x, E_x, atol=1e-12) or np.allclose(R_x.T, E_x, atol=1e-12)
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, z_hat)
    R_y = Gamma_y[:3, :3]
    E_y = expected_R([0, 0, 0], [0, 1, 0], z_hat)
    assert R_y.shape == (3, 3)
    assert np.allclose(R_y, E_y, atol=1e-12) or np.allclose(R_y.T, E_y, atol=1e-12)
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None)
    R_z = Gamma_z[:3, :3]
    E_z = expected_R([0, 0, 0], [0, 0, 1], y_hat)
    assert R_z.shape == (3, 3)
    assert np.allclose(R_z, E_z, atol=1e-12) or np.allclose(R_z.T, E_z, atol=1e-12)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Test specific examples with known behavior and verifies that the tranformation is correct.
    """
    x1, y1, z1 = (1.2, -3.0, 2.5)
    x2, y2, z2 = (-0.8, 4.0, 0.5)
    Gamma = fcn(x1, y1, z1, x2, y2, z2, None)
    assert Gamma.shape == (12, 12)
    R = Gamma[:3, :3]
    assert R.shape == (3, 3)
    I3 = np.eye(3)
    assert np.allclose(R.T @ R, I3, atol=1e-12)
    assert np.allclose(R @ R.T, I3, atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    for i in range(4):
        bi = slice(3 * i, 3 * (i + 1))
        assert np.allclose(Gamma[bi, bi], R, atol=1e-12)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            bi = slice(3 * i, 3 * (i + 1))
            bj = slice(3 * j, 3 * (j + 1))
            assert np.allclose(Gamma[bi, bj], 0.0, atol=1e-12)
    I12 = np.eye(12)
    assert np.allclose(Gamma.T @ Gamma, I12, atol=1e-12)
    assert np.allclose(Gamma @ Gamma.T, I12, atol=1e-12)
    Gamma_x = fcn(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(Gamma_x, I12, atol=1e-12)
    u_local = np.zeros(12)
    v_local = np.array([0.0, 5.0, 0.0])
    u_local[6:9] = v_local
    expected_global = np.zeros(12)
    expected_global[6:9] = R @ v_local
    cond1 = np.allclose(Gamma @ u_local, expected_global, atol=1e-12)
    cond2 = np.allclose(Gamma.T @ u_local, expected_global, atol=1e-12)
    assert cond1 or cond2
    u_local_rot = np.zeros(12)
    w_local = np.array([0.0, 0.0, 3.0])
    u_local_rot[3:6] = w_local
    expected_global_rot = np.zeros(12)
    expected_global_rot[3:6] = R @ w_local
    cond3 = np.allclose(Gamma @ u_local_rot, expected_global_rot, atol=1e-12)
    cond4 = np.allclose(Gamma.T @ u_local_rot, expected_global_rot, atol=1e-12)
    assert cond3 or cond4
    Gv = fcn(0.0, 1.0, 2.0, 0.0, 1.0, 5.0, None)
    Rv = Gv[:3, :3]
    e_z = np.array([0.0, 0.0, 1.0])
    assert np.allclose(Rv[:, 0], e_z, atol=1e-12) or np.allclose(Rv[0, :], e_z, atol=1e-12)
    assert np.isclose(np.linalg.det(Rv), 1.0, atol=1e-12)

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises appropriate ValueError exceptions for invalid reference vectors.
    Verifies proper error handling for:
    1. Non-unit vector: When the reference vector's magnitude is not equal to 1.0
    2. Parallel reference vector: When the reference vector is parallel to the beam axis
    3. Zero-length beam: start and end coordinates are identical
    All conditions should raise ValueError.
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, -1.0, 3.0, 1.0, -1.0, 3.0, None)