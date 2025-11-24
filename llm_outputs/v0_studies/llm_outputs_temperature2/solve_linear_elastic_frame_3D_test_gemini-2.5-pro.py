def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 10.0
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = I_y + I_z
    P_mag = 1000.0
    n_elem = 10
    n_nodes = n_elem + 1
    beam_axis_vec = np.array([1.0, 1.0, 1.0])
    beam_axis_norm = beam_axis_vec / np.linalg.norm(beam_axis_vec)
    node_coords = np.array([i * (L / n_elem) * beam_axis_norm for i in range(n_nodes)])
    local_z_dir_unnorm = np.array([1.0, -1.0, 0.0])
    local_z_dir = local_z_dir_unnorm / np.linalg.norm(local_z_dir_unnorm)
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_dir} for i in range(n_elem)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_vec = P_mag * local_z_dir
    nodal_loads = {n_nodes - 1: list(force_vec) + [0.0, 0.0, 0.0]}
    delta_analytical = P_mag * L ** 3 / (3 * E * I_y)
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node_idx = n_nodes - 1
    tip_disp_vec = u[tip_node_idx * 6:tip_node_idx * 6 + 3]
    delta_calculated = np.linalg.norm(tip_disp_vec)
    assert np.isclose(delta_calculated, delta_analytical, rtol=0.01)
    disp_dir = tip_disp_vec / delta_calculated
    force_dir = force_vec / P_mag
    assert np.allclose(disp_dir, force_dir)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [3.0, 0.0, 5.0], [3.0, 4.0, 5.0]])
    props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}
    elements = [{'node_i': 0, 'node_j': 1, **props, 'local_z': np.array([1.0, 0.0, 0.0])}, {'node_i': 1, 'node_j': 2, **props, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 3, **props, 'local_z': np.array([1.0, 0.0, 0.0])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.allclose(u0, 0.0), 'Displacements should be zero for zero loads'
    assert np.allclose(r0, 0.0), 'Reactions should be zero for zero loads'
    loads1 = {3: [100.0, 200.0, -50.0, 30.0, -60.0, 90.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert not np.allclose(u1, 0.0), 'Displacements should be non-zero for non-zero loads'
    assert not np.allclose(r1, 0.0), 'Reactions should be non-zero for non-zero loads'
    loads2 = {key: [2 * val for val in vec] for (key, vec) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2 * u1), 'Doubling loads should double displacements'
    assert np.allclose(r2, 2 * r1), 'Doubling loads should double reactions'
    loads3 = {key: [-1 * val for val in vec] for (key, vec) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -1 * u1), 'Negating loads should negate displacements'
    assert np.allclose(r3, -1 * r1), 'Negating loads should negate reactions'
    applied_forces = np.array(loads1[3][:3])
    reaction_forces = r1[0:3]
    assert np.allclose(applied_forces + reaction_forces, 0.0, atol=1e-09)
    pos_load = node_coords[3]
    applied_force_moment = np.cross(pos_load, applied_forces)
    applied_moment = np.array(loads1[3][3:])
    total_applied_moment = applied_force_moment + applied_moment
    pos_reaction = node_coords[0]
    reaction_force_moment = np.cross(pos_reaction, reaction_forces)
    reaction_moment = r1[3:6]
    total_reaction_moment = reaction_force_moment + reaction_moment
    assert np.allclose(total_applied_moment + total_reaction_moment, 0.0, atol=1e-09)