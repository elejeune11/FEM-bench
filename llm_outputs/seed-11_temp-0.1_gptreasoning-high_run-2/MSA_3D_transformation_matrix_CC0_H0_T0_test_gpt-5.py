def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    Cases:
    """
    atol = 1e-12
    Gamma_x = fcn(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    assert Gamma_x.shape == (12, 12)
    R_x = Gamma_x[0:3, 0:3]
    R_x_expected = np.eye(3)
    assert np.allclose(R_x, R_x_expected, atol=atol)
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 7.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_y = Gamma_y[0:3, 0:3]
    R_y_expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R_y, R_y_expected, atol=atol)
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 9.0, None)
    R_z = Gamma_z[0:3, 0:3]
    R_z_expected = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(R_z, R_z_expected, atol=atol)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Verifies orthonormality, block structure, determinant, repeated blocks, and round-trip mapping.
    """
    atol = 1e-12

    def expected_R(p1, p2, ref):
        x = np.array(p2, dtype=float) - np.array(p1, dtype=float)
        L = np.linalg.norm(x)
        assert L > 0
        ex = x / L
        if ref is None:
            if abs(ex[0]) < 1e-14 and abs(ex[1]) < 1e-14:
                ref_vec = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                ref_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            ref_vec = np.array(ref, dtype=float)
        ey_temp = np.cross(ref_vec, ex)
        ey_norm = np.linalg.norm(ey_temp)
        assert ey_norm > 0
        ey = ey_temp / ey_norm
        ez = np.cross(ex, ey)
        ez /= np.linalg.norm(ez)
        return np.vstack((ex, ey, ez))

    def check_gamma(p1, p2, ref):
        Gamma = fcn(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], None if ref is None else np.array(ref))
        assert Gamma.shape == (12, 12)
        R_exp = expected_R(p1, p2, ref)
        blocks = [Gamma[0:3, 0:3], Gamma[3:6, 3:6], Gamma[6:9, 6:9], Gamma[9:12, 9:12]]
        for B in blocks:
            assert np.allclose(B, R_exp, atol=atol)
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                sub = Gamma[3 * i:3 * i + 3, 3 * j:3 * j + 3]
                assert np.allclose(sub, np.zeros((3, 3)), atol=atol)
        I3 = np.eye(3)
        assert np.allclose(R_exp.T @ R_exp, I3, atol=atol)
        assert np.allclose(R_exp @ R_exp.T, I3, atol=atol)
        detR = np.linalg.det(R_exp)
        assert np.isclose(detR, 1.0, atol=1e-12)
        I12 = np.eye(12)
        assert np.allclose(Gamma.T @ Gamma, I12, atol=atol)
        rng = np.random.default_rng(12345)
        g = rng.standard_normal(12)
        local = Gamma @ g
        back = Gamma.T @ local
        assert np.allclose(back, g, atol=1e-12)
    check_gamma((0.0, 0.0, 0.0), (2.0, 3.0, 4.0), None)
    check_gamma((1.0, 2.0, 3.0), (1.0, 2.0, 8.0), None)
    check_gamma((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0))

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that the function raises ValueError for:
    1) Non-unit reference vector
    2) Reference vector parallel to beam axis
    3) Zero-length beam
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None)