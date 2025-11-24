def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.012
    L = 2.5
    Iy = 7.5e-06
    Iz = 1.2e-05
    J = 9e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    evals = np.linalg.eigvalsh(K)
    scale = max(E * A / L, E * max(Iy, Iz) / L)
    zeros = np.sum(np.abs(evals) < 1e-09 * scale)
    assert zeros >= 6
    assert np.all(evals >= -1e-09 * scale)
    EA_L = E * A / L
    G = E / (2.0 * (1.0 + nu))
    GJ_L = G * J / L
    assert np.isclose(K[0, 0], EA_L, rtol=1e-12, atol=1e-12)
    assert np.isclose(K[0, 6], -EA_L, rtol=1e-12, atol=1e-12)
    assert np.isclose(K[6, 6], EA_L, rtol=1e-12, atol=1e-12)
    assert np.isclose(K[3, 3], GJ_L, rtol=1e-12, atol=1e-12)
    assert np.isclose(K[3, 9], -GJ_L, rtol=1e-12, atol=1e-12)
    assert np.isclose(K[9, 9], GJ_L, rtol=1e-12, atol=1e-12)
    EIz = E * Iz
    kv = np.array([[12 * EIz / L ** 3, 6 * EIz / L ** 2, -12 * EIz / L ** 3, 6 * EIz / L ** 2], [6 * EIz / L ** 2, 4 * EIz / L, -6 * EIz / L ** 2, 2 * EIz / L], [-12 * EIz / L ** 3, -6 * EIz / L ** 2, 12 * EIz / L ** 3, -6 * EIz / L ** 2], [6 * EIz / L ** 2, 2 * EIz / L, -6 * EIz / L ** 2, 4 * EIz / L]])
    idx_v = [1, 5, 7, 11]
    assert np.allclose(K[np.ix_(idx_v, idx_v)], kv, rtol=1e-12, atol=1e-09)
    EIy = E * Iy
    kw = np.array([[12 * EIy / L ** 3, 6 * EIy / L ** 2, -12 * EIy / L ** 3, 6 * EIy / L ** 2], [6 * EIy / L ** 2, 4 * EIy / L, -6 * EIy / L ** 2, 2 * EIy / L], [-12 * EIy / L ** 3, -6 * EIy / L ** 2, 12 * EIy / L ** 3, -6 * EIy / L ** 2], [6 * EIy / L ** 2, 2 * EIy / L, -6 * EIy / L ** 2, 4 * EIy / L]])
    idx_w = [2, 4, 8, 10]
    assert np.allclose(K[np.ix_(idx_w, idx_w)], kw, rtol=1e-12, atol=1e-09)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Verify tip deflection of a single-element cantilever matches Euler-Bernoulli theory:
    """
    E = 210000000000.0
    nu = 0.29
    A = 0.008
    L = 3.0
    Iy = 6e-06
    Iz = 9e-06
    J = 8e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K22 = K[6:, 6:]
    P = 1000.0
    f2 = np.zeros(6)
    f2[2] = P
    d2 = np.linalg.solve(K22, f2)
    w_tip_expected = P * L ** 3 / (3 * E * Iy)
    assert np.isclose(d2[2], w_tip_expected, rtol=1e-09, atol=1e-12)
    f2 = np.zeros(6)
    f2[1] = P
    d2 = np.linalg.solve(K22, f2)
    v_tip_expected = P * L ** 3 / (3 * E * Iz)
    assert np.isclose(d2[1], v_tip_expected, rtol=1e-09, atol=1e-12)
    f2 = np.zeros(6)
    f2[0] = P
    d2 = np.linalg.solve(K22, f2)
    u_tip_expected = P * L / (E * A)
    assert np.isclose(d2[0], u_tip_expected, rtol=1e-12, atol=1e-15)