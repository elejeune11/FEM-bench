def test_cardinal_axis_alignment(fcn):
    """
    Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the first 3x3 direction cosine matrix has the expected orientation for beams aligned with:
    """
    Gamma_x = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_x = Gamma_x[:3, :3]
    R_x_expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R_x, R_x_expected)
    Gamma_y = fcn(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.array([0.0, 0.0, 1.0]))
    R_y = Gamma_y[:3, :3]
    R_y_expected = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R_y, R_y_expected)
    Gamma_z = fcn(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, None)
    R_z = Gamma_z[:3, :3]
    R_z_expected = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(R_z, R_z_expected)

def test_transformation_matrix_properties(fcn):
    """
    Test fundamental properties of the transformation matrices:
    """
    Gamma = fcn(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, None)
    assert Gamma.shape == (12, 12)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    R_expected = np.array([[inv_sqrt2, inv_sqrt2, 0.0], [-inv_sqrt2, inv_sqrt2, 0.0], [0.0, 0.0, 1.0]])
    R = Gamma[:3, :3]
    assert np.allclose(R, R_expected, atol=1e-12)
    I12 = np.eye(12)
    assert np.allclose(Gamma.T @ Gamma, I12, atol=1e-12, rtol=1e-12)
    rng = np.random.default_rng(0)
    v_global = rng.standard_normal(12)
    v_local = Gamma @ v_global
    v_global_back = Gamma.T @ v_local
    assert np.allclose(v_global_back, v_global, atol=1e-12, rtol=1e-12)
    for i in range(4):
        block = Gamma[3 * i:3 * (i + 1), 0:3]
        assert np.allclose(block, R_expected, atol=1e-12)
    for i in range(4):
        for j in range(4):
            block = Gamma[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
            if i == j:
                assert np.allclose(block, R_expected, atol=1e-12)
            else:
                assert np.allclose(block, np.zeros((3, 3)), atol=1e-12)
    Gamma_alt = fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 1.0, 0.0]))
    R_alt = Gamma_alt[:3, :3]
    R_alt_expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert np.allclose(R_alt, R_alt_expected, atol=1e-12)
    assert np.allclose(Gamma_alt.T @ Gamma_alt, I12, atol=1e-12, rtol=1e-12)

def test_beam_transformation_matrix_error_messages(fcn):
    """
    Test that ValueError is raised for:
    1) Non-unit reference vector
    2) Reference vector parallel to the beam axis
    3) Zero-length beam (start and end nodes coincide)
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, None)