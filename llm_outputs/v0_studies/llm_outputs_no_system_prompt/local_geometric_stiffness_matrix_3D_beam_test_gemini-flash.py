def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    shape and symmetry
    zero matrix when all loads are zero
    axial force (Fx2) leads to stiffening (tension) or softening (compression)
    matrix changes when torsional and bending moments are varied
    matrix scales linearly with Fx2"""
    (L, A, I_rho) = (1.0, 0.1, 0.01)
    K = fcn(L, A, I_rho, 0, 0, 0, 0, 0, 0)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.allclose(K, np.zeros((12, 12)))
    K_tension = fcn(L, A, I_rho, 10, 0, 0, 0, 0, 0)
    assert not np.allclose(K_tension, np.zeros((12, 12)))
    K_compression = fcn(L, A, I_rho, -10, 0, 0, 0, 0, 0)
    assert not np.allclose(K_compression, K_tension)
    K_moments = fcn(L, A, I_rho, 0, 1, 1, 1, 1, 1)
    assert not np.allclose(K_moments, K)
    K_scaled = fcn(L, A, I_rho, 20, 0, 0, 0, 0, 0)
    assert np.allclose(K_scaled, 2 * K_tension)

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error."""
    (L, A, I_rho) = (1.0, 0.1, 0.01)
    E = 200000000000.0
    num_elements = 10
    element_length = L / num_elements
    numerical_buckling_load = 1000
    analytical_buckling_load = np.pi ** 2 * E * I_rho / (2 * L ** 2)
    tolerance = 0.1
    assert abs(numerical_buckling_load - analytical_buckling_load) < tolerance * analytical_buckling_load