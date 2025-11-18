def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.0
    A = 1.0
    I_rho = 1.0
    Kg_zero = fcn(L, A, I_rho, Fx2=0.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert isinstance(Kg_zero, np.ndarray)
    assert Kg_zero.shape == (12, 12)
    assert np.allclose(Kg_zero, Kg_zero.T, rtol=1e-12, atol=1e-12)
    assert np.allclose(Kg_zero, np.zeros((12, 12)), rtol=0, atol=1e-14)
    Kg_tension = fcn(L, A, I_rho, Fx2=100.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    Kg_compression = fcn(L, A, I_rho, Fx2=-100.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    x = np.zeros(12)
    x[1] = 0.3
    x[2] = -0.2
    x[4] = 0.1
    x[5] = -0.4
    x[7] = -0.15
    x[8] = 0.25
    x[10] = -0.05
    x[11] = 0.2
    qt = float(x.T @ Kg_tension @ x)
    qc = float(x.T @ Kg_compression @ x)
    assert qt > 0.0
    assert qc < 0.0
    Kg_mom1 = fcn(L, A, I_rho, Fx2=0.0, Mx2=150.0, My1=50.0, Mz1=-30.0, My2=-20.0, Mz2=40.0)
    Kg_mom2 = fcn(L, A, I_rho, Fx2=0.0, Mx2=-75.0, My1=-10.0, Mz1=15.0, My2=60.0, Mz2=-25.0)
    assert not np.allclose(Kg_mom1, np.zeros((12, 12)), atol=1e-14)
    assert not np.allclose(Kg_mom1, Kg_mom2, rtol=1e-06, atol=1e-09)
    Kg_unit = fcn(L, A, I_rho, Fx2=1.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    scale = 3.5
    Kg_scaled = fcn(L, A, I_rho, Fx2=scale, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert np.allclose(Kg_scaled, scale * Kg_unit, rtol=1e-12, atol=1e-12)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to
    the correct buckling load for a cantilever column. Compare the smallest
    eigenvalue-based buckling load from a single finite element to the analytical
    Euler buckling load Pcr = π^2 EI / (4 L^2). Use tolerances that account for
    discretization error with one element.
    """
    L = 2.0
    E = 210000000000.0
    nu = 0.3
    G = E / (2 * (1 + nu))
    A = 0.01
    I = 8e-06
    Iy = I
    Iz = I
    J = 5e-06
    Ke = np.zeros((12, 12))
    k_ax = E * A / L
    Ke[0, 0] += k_ax
    Ke[0, 6] -= k_ax
    Ke[6, 0] -= k_ax
    Ke[6, 6] += k_ax
    k_t = G * J / L
    Ke[3, 3] += k_t
    Ke[3, 9] -= k_t
    Ke[9, 3] -= k_t
    Ke[9, 9] += k_t
    fz = E * Iz / L ** 3
    Kbz = fz * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]])
    dofz = [1, 5, 7, 11]
    for i in range(4):
        for j in range(4):
            Ke[dofz[i], dofz[j]] += Kbz[i, j]
    fy = E * Iy / L ** 3
    Kby = fy * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]])
    dofy = [2, 4, 8, 10]
    for i in range(4):
        for j in range(4):
            Ke[dofy[i], dofy[j]] += Kby[i, j]
    Kg_unit = fcn(L, A=1.0, I_rho=1.0, Fx2=-1.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    free = np.arange(6, 12)
    Ke_ff = Ke[np.ix_(free, free)]
    Kg_ff = Kg_unit[np.ix_(free, free)]
    M = -np.linalg.inv(Ke_ff) @ Kg_ff
    evals = np.linalg.eigvals(M)
    evals = np.array([ev.real for ev in evals if abs(ev.imag) < 1e-08 * max(1.0, abs(ev.real))])
    evals = evals[evals > 0.0]
    assert evals.size > 0
    Pcr_FE = float(np.min(evals))
    Pcr_exact = np.pi ** 2 * E * I / (4.0 * L ** 2)
    rel_err = abs(Pcr_FE - Pcr_exact) / Pcr_exact
    assert rel_err < 0.15