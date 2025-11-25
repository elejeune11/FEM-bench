def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 3.7
    A = 0.02
    I_rho = 5e-06
    K0 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, atol=1e-12)
    assert np.allclose(K0, 0.0, atol=1e-14)
    Fx_mag = 1.234
    K_tension = fcn(L, A, I_rho, Fx_mag, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_compression = fcn(L, A, I_rho, -Fx_mag, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert K_tension.shape == (12, 12)
    assert np.allclose(K_tension, K_tension.T, atol=1e-12)
    assert np.allclose(K_compression, K_compression.T, atol=1e-12)
    assert np.allclose(K_compression, -K_tension, rtol=1e-12, atol=1e-14)
    lateral_dofs = [1, 2, 4, 5, 7, 8, 10, 11]
    sum_diag_tension = float(np.sum(np.diag(K_tension)[lateral_dofs]))
    sum_diag_compression = float(np.sum(np.diag(K_compression)[lateral_dofs]))
    assert sum_diag_tension > 0.0
    assert sum_diag_compression < 0.0
    Fx1 = 1.0
    Fx2 = 2.5
    K1 = fcn(L, A, I_rho, Fx1, 0.0, 0.0, 0.0, 0.0, 0.0)
    K2 = fcn(L, A, I_rho, Fx2, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K2, Fx2 / Fx1 * K1, rtol=1e-12, atol=1e-12)
    K_fx_only = fcn(L, A, I_rho, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_with_moments = fcn(L, A, I_rho, 0.5, 0.7, 1.0, -0.5, -0.8, 0.9)
    assert np.allclose(K_with_moments, K_with_moments.T, atol=1e-12)
    diff_norm = np.linalg.norm(K_with_moments - K_fx_only)
    assert diff_norm > 1e-10

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it predicts
    the correct Euler buckling load for a cantilever column using a single 3D beam element.
    The computed critical load is compared to the analytical value with a tolerance
    that accounts for single-element discretization error.
    """
    L = 1.0
    E = 1.0
    G = 1.0
    A = 1.0
    Iy = 1.0
    Iz = 1.0
    J = 1.0
    I_rho = 1.0
    Ke = np.zeros((12, 12), dtype=float)
    (u1, v1, w1, tx1, ty1, tz1, u2, v2, w2, tx2, ty2, tz2) = range(12)
    k_ax = E * A / L
    Ke[u1, u1] += k_ax
    Ke[u1, u2] -= k_ax
    Ke[u2, u1] -= k_ax
    Ke[u2, u2] += k_ax
    k_tor = G * J / L
    Ke[tx1, tx1] += k_tor
    Ke[tx1, tx2] -= k_tor
    Ke[tx2, tx1] -= k_tor
    Ke[tx2, tx2] += k_tor
    c = E * Iz
    Ke[v1, v1] += 12 * c / L ** 3
    Ke[v1, tz1] += 6 * c / L ** 2
    Ke[v1, v2] += -12 * c / L ** 3
    Ke[v1, tz2] += 6 * c / L ** 2
    Ke[tz1, v1] += 6 * c / L ** 2
    Ke[tz1, tz1] += 4 * c / L
    Ke[tz1, v2] += -6 * c / L ** 2
    Ke[tz1, tz2] += 2 * c / L
    Ke[v2, v1] += -12 * c / L ** 3
    Ke[v2, tz1] += -6 * c / L ** 2
    Ke[v2, v2] += 12 * c / L ** 3
    Ke[v2, tz2] += -6 * c / L ** 2
    Ke[tz2, v1] += 6 * c / L ** 2
    Ke[tz2, tz1] += 2 * c / L
    Ke[tz2, v2] += -6 * c / L ** 2
    Ke[tz2, tz2] += 4 * c / L
    c = E * Iy
    Ke[w1, w1] += 12 * c / L ** 3
    Ke[w1, ty1] += -6 * c / L ** 2
    Ke[w1, w2] += -12 * c / L ** 3
    Ke[w1, ty2] += -6 * c / L ** 2
    Ke[ty1, w1] += -6 * c / L ** 2
    Ke[ty1, ty1] += 4 * c / L
    Ke[ty1, w2] += 6 * c / L ** 2
    Ke[ty1, ty2] += 2 * c / L
    Ke[w2, w1] += -12 * c / L ** 3
    Ke[w2, ty1] += 6 * c / L ** 2
    Ke[w2, w2] += 12 * c / L ** 3
    Ke[w2, ty2] += 6 * c / L ** 2
    Ke[ty2, w1] += -6 * c / L ** 2
    Ke[ty2, ty1] += 2 * c / L
    Ke[ty2, w2] += 6 * c / L ** 2
    Ke[ty2, ty2] += 4 * c / L
    Kg_unit = fcn(L, A, I_rho, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    free = [v2, w2, ty2, tz2]
    Ke_red = Ke[np.ix_(free, free)]
    Kg_red = Kg_unit[np.ix_(free, free)]
    M = np.linalg.solve(Kg_red, Ke_red)
    eigvals = np.linalg.eigvals(M)
    eigvals = np.real(eigvals[np.isclose(np.imag(eigvals), 0.0, atol=1e-09)])
    eigvals = eigvals[eigvals > 0]
    Pcr_numeric = float(np.min(eigvals))
    Pcr_analytical = np.pi ** 2 * E * Iy / (4.0 * L ** 2)
    assert np.isclose(Pcr_numeric, Pcr_analytical, rtol=0.03, atol=0.0)