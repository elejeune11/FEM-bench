def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the first 3x3 direction cosine submatrix has the expected orientation
    for beams aligned with each cardinal direction using the specified reference vector logic.
    """
    Gamma_x = fcn(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_x = Gamma_x[:3, :3]
    expected_R_x = np.eye(3, dtype=float)
    assert R_x.shape == (3, 3)
    assert np.allclose(R_x, expected_R_x, atol=1e-12)
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 7.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_y = Gamma_y[:3, :3]
    expected_R_y = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    assert R_y.shape == (3, 3)
    assert np.allclose(R_y, expected_R_y, atol=1e-12)
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 2.0, None)
    R_z = Gamma_z[:3, :3]
    expected_R_z = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    assert R_z.shape == (3, 3)
    assert np.allclose(R_z, expected_R_z, atol=1e-12)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of the 12x12 transformation matrix.
    Verifies:
    """
    test_cases = [((1.0, 2.0, 3.0, 4.0, 6.0, 9.0, None), 'non-vertical_default_z'), ((-2.0, 1.0, 0.5, 1.0, -2.0, 2.0, np.array([0.0, 1.0, 0.0])), 'explicit_unit_ref'), ((5.0, -1.0, 1.0, 5.0, -1.0, 10.0, None), 'vertical_default_y')]
    I3 = np.eye(3)
    I12 = np.eye(12)
    for (params, _) in test_cases:
        (x1, y1, z1, x2, y2, z2, ref) = params
        Gamma = fcn(x1, y1, z1, x2, y2, z2, ref)
        assert Gamma.shape == (12, 12)
        R = Gamma[0:3, 0:3]
        for i in range(4):
            blk = Gamma[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3]
            assert np.allclose(blk, R, atol=1e-12)
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                off_blk = Gamma[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                assert np.allclose(off_blk, 0.0, atol=1e-12)
        assert np.allclose(R @ R.T, I3, atol=1e-12)
        assert np.allclose(R.T @ R, I3, atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
        assert np.allclose(Gamma.T @ Gamma, I12, atol=1e-12)
        (dx, dy, dz) = (x2 - x1, y2 - y1, z2 - z1)
        v = np.array([dx, dy, dz], dtype=float)
        v_norm = np.linalg.norm(v)
        assert v_norm > 0.0
        x_local = v / v_norm
        if ref is None:
            k = np.array([0.0, 0.0, 1.0], dtype=float)
            j = np.array([0.0, 1.0, 0.0], dtype=float)
            if np.isclose(abs(np.dot(x_local, k)), 1.0, atol=1e-12):
                ref_vec = j
            else:
                ref_vec = k
        else:
            ref_vec = np.array(ref, dtype=float)
        y_local = np.cross(ref_vec, x_local)
        y_norm = np.linalg.norm(y_local)
        assert y_norm > 0.0
        y_local /= y_norm
        z_local = np.cross(x_local, y_local)
        expected_R = np.vstack([x_local, y_local, z_local])
        assert np.allclose(R, expected_R, atol=1e-12)

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that ValueError is raised for invalid inputs:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 2.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 0.0, 5.0, 0.0, np.array([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, None)