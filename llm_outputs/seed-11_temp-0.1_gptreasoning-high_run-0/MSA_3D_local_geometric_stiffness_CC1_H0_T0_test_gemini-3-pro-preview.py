def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 4.0
    A = 0.05
    I_rho = 0.001
    kg_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert kg_zero.shape == (12, 12)
    assert np.allclose(kg_zero, 0.0), 'Matrix should be zero when no loads are present.'
    kg_loaded = fcn(L, A, I_rho, -1000.0, 50.0, 20.0, -10.0, -20.0, 10.0)
    assert np.allclose(kg_loaded, kg_loaded.T), 'Geometric stiffness matrix must be symmetric.'
    kg_tension = fcn(L, A, I_rho, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    kg_compression = fcn(L, A, I_rho, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(kg_tension, -kg_compression), 'Geometric stiffness should flip sign when axial force direction reverses.'
    assert kg_tension[1, 1] > 0, 'Tension should contribute positive geometric stiffness.'
    assert kg_compression[1, 1] < 0, 'Compression should contribute negative geometric stiffness.'
    kg_tension_double = fcn(L, A, I_rho, 2000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(kg_tension_double, 2.0 * kg_tension), 'Matrix should scale linearly with axial force.'
    kg_base = fcn(L, A, I_rho, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    kg_torsion = fcn(L, A, I_rho, 100.0, 50.0, 0.0, 0.0, 0.0, 0.0)
    kg_bending = fcn(L, A, I_rho, 100.0, 0.0, 50.0, 0.0, 0.0, 0.0)
    assert not np.allclose(kg_base, kg_torsion), 'Matrix should account for torsional moment.'
    assert not np.allclose(kg_base, kg_bending), 'Matrix should account for bending moments.'

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    L = 10.0
    E = 200000000000.0
    I = 5e-05
    A = 0.01
    I_rho = 0.0001
    P_exact = np.pi ** 2 * E * I / (4 * L ** 2)
    k_vv = 12 * E * I / L ** 3
    k_vt = -6 * E * I / L ** 2
    k_tt = 4 * E * I / L
    Ke_reduced = np.array([[k_vv, k_vt], [k_vt, k_tt]])
    Kg_full = fcn(L, A, I_rho, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_reduced = np.array([[Kg_full[7, 7], Kg_full[7, 11]], [Kg_full[11, 7], Kg_full[11, 11]]])
    vals = eigvals(Ke_reduced, -Kg_reduced)
    buckling_loads = [v.real for v in vals if np.isreal(v) and v.real > 0]
    assert len(buckling_loads) > 0, 'No positive buckling load found.'
    P_numerical = min(buckling_loads)
    assert np.isclose(P_numerical, P_exact, rtol=0.25), f'Numerical buckling load {P_numerical} deviates too much from analytical {P_exact}.'