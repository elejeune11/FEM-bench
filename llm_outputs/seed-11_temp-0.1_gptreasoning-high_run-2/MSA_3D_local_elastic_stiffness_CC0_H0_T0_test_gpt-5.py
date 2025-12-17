def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.02
    L = 3.0
    Iy = 4e-06
    Iz = 7e-06
    J = 5e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert isinstance(K, np.ndarray)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    vals = np.linalg.eigvalsh(K)
    scale = max(np.max(np.abs(np.diag(K))), 1.0)
    tol = 1e-08 * scale
    zero_count = np.sum(np.abs(vals) < tol)
    assert zero_count >= 6
    k_axial = E * A / L
    idx_axial = [0, 6]
    K_axial = K[np.ix_(idx_axial, idx_axial)]
    K_axial_exp = k_axial * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_axial, K_axial_exp, rtol=1e-12, atol=1e-12)
    G = E / (2.0 * (1.0 + nu))
    k_torsion = G * J / L
    idx_torsion = [3, 9]
    K_torsion = K[np.ix_(idx_torsion, idx_torsion)]
    K_torsion_exp = k_torsion * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(K_torsion, K_torsion_exp, rtol=1e-12, atol=1e-12)
    factor_z = E * Iz / L ** 3
    L2 = L ** 2
    idx_bz = [1, 5, 7, 11]
    K_bz = K[np.ix_(idx_bz, idx_bz)]
    K_bz_exp = factor_z * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    assert np.allclose(K_bz, K_bz_exp, rtol=1e-12, atol=1e-12)
    factor_y = E * Iy / L ** 3
    idx_by = [2, 4, 8, 10]
    K_by = K[np.ix_(idx_by, idx_by)]
    K_by_exp = factor_y * np.array([[12.0, -6.0 * L, -12.0, -6.0 * L], [-6.0 * L, 4.0 * L2, 6.0 * L, 2.0 * L2], [-12.0, 6.0 * L, 12.0, 6.0 * L], [-6.0 * L, 2.0 * L2, 6.0 * L, 4.0 * L2]])
    assert np.allclose(K_by, K_by_exp, rtol=1e-12, atol=1e-12)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """
    Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.29
    A = 0.015
    L = 2.0
    Iy = 7.5e-06
    Iz = 1.2e-05
    J = 8e-06
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K22 = K[6:, 6:]
    Pz = 1000.0
    Fz = np.zeros(6)
    Fz[2] = Pz
    d2_z = np.linalg.solve(K22, Fz)
    w2 = d2_z[2]
    w2_theory = Pz * L ** 3 / (3.0 * E * Iy)
    assert np.isclose(w2, w2_theory, rtol=1e-10, atol=1e-12)
    Py = 800.0
    Fy = np.zeros(6)
    Fy[1] = Py
    d2_y = np.linalg.solve(K22, Fy)
    v2 = d2_y[1]
    v2_theory = Py * L ** 3 / (3.0 * E * Iz)
    assert np.isclose(v2, v2_theory, rtol=1e-10, atol=1e-12)
    Px = 500.0
    Fx = np.zeros(6)
    Fx[0] = Px
    d2_x = np.linalg.solve(K22, Fx)
    u2 = d2_x[0]
    u2_theory = Px * L / (E * A)
    assert np.isclose(u2, u2_theory, rtol=1e-12, atol=1e-15)