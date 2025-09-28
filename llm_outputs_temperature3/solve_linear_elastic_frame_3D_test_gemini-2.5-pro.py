def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 1.0
    n_elem = 10
    n_nodes = n_elem + 1
    E = 210000000000.0
    nu = 0.3
    b = 0.1
    A = b ** 2
    I = b ** 4 / 12
    J = 0.1406 * b ** 4
    P_mag = 1000.0
    beam_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    local_z_dir = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    local_y_dir = np.cross(local_z_dir, beam_axis)
    node_coords = np.array([i * (L / n_elem) * beam_axis for i in range(n_nodes)])
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z_dir})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    load_vec = P_mag * local_y_dir
    nodal_loads = {n_nodes - 1: np.hstack([load_vec, [0.0, 0.0, 0.0]])}
    analytical_deflection = P_mag * L ** 3 / (3 * E * I)
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_displacement_vec = u[(n_nodes - 1) * 6:(n_nodes - 1) * 6 + 3]
    numerical_deflection = np.linalg.norm(tip_displacement_vec)
    disp_dir = tip_displacement_vec / numerical_deflection
    assert np.allclose(disp_dir, local_y_dir)
    assert numerical_deflection == pytest.approx(analytical_deflection, rel=0.01)

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-06
    I_z = 8.333e-06
    J = 1.406e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0, 0, 1]}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.all(u_zero == 0.0)
    assert np.all(r_zero == 0.0)
    base_loads = {2: np.array([1000.0, 2000.0, 3000.0, 400.0, 500.0, 600.0])}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, base_loads)
    assert not np.all(u1[6:] == 0.0)
    assert not np.all(r1[0:6] == 0.0)
    assert np.all(u1[0:6] == 0.0)
    assert np.all(r1[6:] == 0.0)
    doubled_loads = {k: 2 * v for (k, v) in base_loads.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, doubled_loads)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    negated_loads = {k: -v for (k, v) in base_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, negated_loads)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    applied_forces = base_loads[2][:3]
    reaction_forces = r1[0:6][:3]
    assert np.allclose(applied_forces + reaction_forces, np.zeros(3), atol=1e-09)
    applied_moments_at_node = base_loads[2][3:]
    moment_from_applied_forces = np.cross(node_coords[2], applied_forces)
    total_applied_moment = applied_moments_at_node + moment_from_applied_forces
    reaction_moments = r1[0:6][3:]
    assert np.allclose(total_applied_moment + reaction_moments, np.zeros(3), atol=1e-09)