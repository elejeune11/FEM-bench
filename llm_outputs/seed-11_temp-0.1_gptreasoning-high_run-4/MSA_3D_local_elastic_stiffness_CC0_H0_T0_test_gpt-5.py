def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.29
    A = 0.013
    L = 2.4
    Iy = 5.1e-06
    Iz = 8.4e-06
    J = 1.7e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    s = np.linalg.svd(K, compute_uv=False)
    tol = 1e-10 * s.max()
    rank = np.sum(s > tol)
    assert rank == 6
    assert 12 - rank == 6
    EA_L = E * A / L
    idx_ax = [0, 6]
    k_ax = K[np.ix_(idx_ax, idx_ax)]
    expected_ax = EA_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k_ax, expected_ax, rtol=1e-12, atol=1e-12)
    off_ax = K[np.ix_(idx_ax, [i for i in range(12) if i not in idx_ax])]
    assert np.allclose(off_ax, 0.0, atol=1e-12)
    G = E / (2.0 * (1.0 + nu))
    GJ_L = G * J / L
    idx_tor = [3, 9]
    k_tor = K[np.ix_(idx_tor, idx_tor)]
    expected_tor = GJ_L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k_tor, expected_tor, rtol=1e-12, atol=1e-12)
    off_tor = K[np.ix_(idx_tor, [i for i in range(12) if i not in idx_tor])]
    assert np.allclose(off_tor, 0.0, atol=1e-12)
    idx_v = [1, 5, 7, 11]
    k_v = K[np.ix_(idx_v, idx_v)]
    EIz = E * Iz
    L2 = L * L
    L3 = L2 * L
    expected_v = EIz / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    assert np.allclose(k_v, expected_v, rtol=1e-12, atol=1e-12)
    idx_w = [2, 4, 8, 10]
    k_w = K[np.ix_(idx_w, idx_w)]
    EIy = E * Iy
    expected_w = EIy / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    assert np.allclose(k_w, expected_w, rtol=1e-12, atol=1e-12)
    cross_vw = K[np.ix_(idx_v, idx_w)]
    assert np.allclose(cross_vw, 0.0, atol=1e-12)
    off_ax_bend = K[np.ix_([0, 6], idx_v + idx_w)]
    off_tor_bend = K[np.ix_([3, 9], idx_v + idx_w)]
    assert np.allclose(off_ax_bend, 0.0, atol=1e-12)
    assert np.allclose(off_tor_bend, 0.0, atol=1e-12)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    L = 2.5
    Iy = 6e-06
    Iz = 9e-06
    J = 1.8e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    free = np.array([6, 7, 8, 9, 10, 11], dtype=int)
    K_ff = K[np.ix_(free, free)]
    Fz = 1000.0
    f = np.zeros(6)
    f[2] = Fz
    d = np.linalg.solve(K_ff, f)
    w_tip = d[2]
    w_exact = Fz * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(w_tip, w_exact, rtol=1e-12, atol=1e-12)
    Fy = 800.0
    f = np.zeros(6)
    f[1] = Fy
    d = np.linalg.solve(K_ff, f)
    v_tip = d[1]
    v_exact = Fy * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(v_tip, v_exact, rtol=1e-12, atol=1e-12)
    Fx = 5000.0
    f = np.zeros(6)
    f[0] = Fx
    d = np.linalg.solve(K_ff, f)
    u_tip = d[0]
    u_exact = Fx * L / (E * A)
    assert np.isclose(u_tip, u_exact, rtol=1e-12, atol=1e-12)