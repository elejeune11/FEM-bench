def test_local_stiffness_3D_beam(fcn):
    """Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    shape check
    symmetry
    expected singularity due to rigid body modes
    block-level verification of axial, torsion, and bending terms"""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 5.0
    Iy = 8.333e-06
    Iz = 1.666e-05
    J = 1.5e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    eigenvalues = np.linalg.eigvalsh(K)
    zero_eigenvalues = np.sum(np.abs(eigenvalues) < 1e-10)
    assert zero_eigenvalues == 6
    axial_stiffness = E * A / L
    assert np.isclose(K[0, 0], axial_stiffness)
    assert np.isclose(K[0, 6], -axial_stiffness)
    torsional_stiffness = E * J / (2 * (1 + nu) * L)
    assert np.isclose(K[3, 3], torsional_stiffness)
    assert np.isclose(K[3, 9], -torsional_stiffness)
    bending_y_stiffness = 12 * E * Iz / L ** 3
    assert np.isclose(K[1, 1], bending_y_stiffness)
    bending_z_stiffness = 12 * E * Iy / L ** 3
    assert np.isclose(K[2, 2], bending_z_stiffness)

def test_cantilever_deflection_matches_euler_bernoulli(fcn):
    """Apply a perpendicular point load in the z direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a perpendicular point load in the y direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory.
    Apply a parallel point load in the x direction to the tip of a cantilever beam and verify that the computed displacement matches the analytical solution from Euler-Bernoulli beam theory."""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    L = 5.0
    Iy = 8.333e-06
    Iz = 1.666e-05
    J = 1.5e-05
    K = fcn(E, nu, A, L, Iy, Iz, J)
    K_reduced = K[6:, 6:]
    F_z = 1000.0
    force_vector_z = np.array([0, 0, F_z, 0, 0, 0])
    displacement_z = np.linalg.solve(K_reduced, force_vector_z)
    analytical_deflection_z = F_z * L ** 3 / (3 * E * Iy)
    assert np.isclose(displacement_z[2], analytical_deflection_z, rtol=0.001)
    F_y = 1000.0
    force_vector_y = np.array([0, F_y, 0, 0, 0, 0])
    displacement_y = np.linalg.solve(K_reduced, force_vector_y)
    analytical_deflection_y = F_y * L ** 3 / (3 * E * Iz)
    assert np.isclose(displacement_y[1], analytical_deflection_y, rtol=0.001)
    F_x = 1000.0
    force_vector_x = np.array([F_x, 0, 0, 0, 0, 0])
    displacement_x = np.linalg.solve(K_reduced, force_vector_x)
    analytical_elongation = F_x * L / (E * A)
    assert np.isclose(displacement_x[0], analytical_elongation, rtol=0.001)