def test_cardinal_axis_alignment(fcn):
    """Test transformation matrices for beams aligned with global coordinate axes.
    Verifies that the 3x3 direction cosine matrix (first block of the 12x12 transformation
    matrix) has the expected orientation for beams aligned with each cardinal direction.
    Expected orientations are based on:
    Test cases:
    """

    def expected_R(p1, p2, ref_vec):
        x_axis = p2 - p1
        x_axis = x_axis / np.linalg.norm(x_axis)
        if ref_vec is None:
            if np.allclose(np.abs(x_axis), np.array([0.0, 0.0, 1.0]), atol=1e-12):
                ref_vec = np.array([0.0, 1.0, 0.0])
            else:
                ref_vec = np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(ref_vec, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        return np.vstack((x_axis, y_axis, z_axis))
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([5.0, 0.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0])
    Gamma = fcn(*p1, *p2, ref)
    R_exp = expected_R(p1, p2, ref)
    assert Gamma.shape == (12, 12)
    assert np.allclose(Gamma[:3, :3], R_exp, atol=1e-12)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 7.0, 0.0])
    ref = np.array([0.0, 0.0, 1.0])
    Gamma = fcn(*p1, *p2, ref)
    R_exp = expected_R(p1, p2, ref)
    assert Gamma.shape == (12, 12)
    assert np.allclose(Gamma[:3, :3], R_exp, atol=1e-12)
    p1 = np.array([1.0, -2.0, 0.0])
    p2 = np.array([1.0, -2.0, 9.0])
    Gamma = fcn(*p1, *p2, None)
    R_exp = expected_R(p1, p2, None)
    assert Gamma.shape == (12, 12)
    assert np.allclose(Gamma[:3, :3], R_exp, atol=1e-12)

def test_transformation_matrix_properties(fcn):
    """Test fundamental mathematical properties of transformation matrices for multiple beam configurations.
    Checks:
    """

    def expected_R(p1, p2, ref_vec):
        x_axis = p2 - p1
        x_axis = x_axis / np.linalg.norm(x_axis)
        if ref_vec is None:
            if np.allclose(np.abs(x_axis), np.array([0.0, 0.0, 1.0]), atol=1e-12):
                ref_vec = np.array([0.0, 1.0, 0.0])
            else:
                ref_vec = np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(ref_vec, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        return np.vstack((x_axis, y_axis, z_axis))
    p1A = np.array([1.0, -1.0, 2.0])
    p2A = np.array([4.0, 1.0, 5.0])
    GammaA = fcn(*p1A, *p2A, None)
    assert GammaA.shape == (12, 12)
    RA = expected_R(p1A, p2A, None)
    IA = np.eye(12)
    assert np.allclose(GammaA.T @ GammaA, IA, atol=1e-12)
    assert np.allclose(RA @ RA.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(RA), 1.0, atol=1e-12)
    blocksA = [GammaA[i:i + 3, j:j + 3] for i in (0, 3, 6, 9) for j in (0, 3, 6, 9)]
    diag_blocksA = [GammaA[k:k + 3, k:k + 3] for k in (0, 3, 6, 9)]
    for B in diag_blocksA:
        assert np.allclose(B, diag_blocksA[0], atol=1e-12)
    for i in range(4):
        for j in range(4):
            if i != j:
                offB = GammaA[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
                assert np.allclose(offB, np.zeros((3, 3)), atol=1e-12)
    v_blocks = [np.array([1.0, 2.0, -1.0]), np.array([-2.0, 0.5, 3.0]), np.array([0.7, -1.3, 4.2]), np.array([5.5, 0.0, -0.6])]
    v = np.concatenate(v_blocks)
    u_local = GammaA @ v
    u_local_expected = np.concatenate([RA @ vb for vb in v_blocks])
    assert np.allclose(u_local, u_local_expected, atol=1e-12)
    D = np.diag([1.0, 2.0, 3.0])
    K_local = np.zeros((12, 12))
    for k in range(4):
        K_local[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = D
    K_global = GammaA.T @ K_local @ GammaA
    K_expected = np.zeros((12, 12))
    RTDR = RA.T @ D @ RA
    for k in range(4):
        K_expected[3 * k:3 * (k + 1), 3 * k:3 * (k + 1)] = RTDR
    assert np.allclose(K_global, K_expected, atol=1e-12)
    p1B = np.array([2.0, 3.0, 1.0])
    p2B = np.array([2.0, 3.0, 9.0])
    GammaB = fcn(*p1B, *p2B, None)
    RB = expected_R(p1B, p2B, None)
    assert GammaB.shape == (12, 12)
    assert np.allclose(GammaB[:3, :3], RB, atol=1e-12)
    assert np.allclose(GammaB.T @ GammaB, np.eye(12), atol=1e-12)
    assert np.isclose(np.linalg.det(GammaB[:3, :3]), 1.0, atol=1e-12)

def test_beam_transformation_matrix_error_messages(fcn):
    """Test that the function raises appropriate ValueError exceptions for invalid inputs.
    Verifies error handling for:
    1. Non-unit reference vector (magnitude != 1.0)
    2. Parallel reference vector to the beam axis
    3. Zero-length beam (identical start and end coordinates)
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(1.0, -1.0, 3.0, 1.0, -1.0, 3.0, np.array([0.0, 0.0, 1.0]))