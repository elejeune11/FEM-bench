def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
Verify beam tip deflection with the appropriate analytical reference solution."""
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-05
    I_z = 8.33e-05
    J = 0.000166
    L = 10.0
    n_elem = 10
    n_nodes = n_elem + 1
    P = 10000.0
    beam_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    node_coords = np.array([i * (L / n_elem) * beam_axis for i in range(n_nodes)])
    aux_vec = np.array([0.0, 0.0, 1.0])
    y_local = np.cross(aux_vec, beam_axis)
    y_local /= np.linalg.norm(y_local)
    z_local = np.cross(beam_axis, y_local)
    z_local /= np.linalg.norm(z_local)
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': z_local})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_vec = P * z_local
    nodal_loads = {n_nodes - 1: np.concatenate((force_vec, [0.0, 0.0, 0.0]))}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = P * L ** 3 / (3 * E * I_y)
    u_analytical_vec = delta_analytical * z_local
    tip_node_idx = n_nodes - 1
    u_computed_vec = u[tip_node_idx * 6:tip_node_idx * 6 + 3]
    assert np.allclose(u_computed_vec, u_analytical_vec, rtol=0.01)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
Suggested Test stages:
1. Zero loads -> All displacements and reactions should be zero.
2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
3. Double the loads -> Displacements and reactions should double (linearity check).
4. Negate the original loads -> Displacements and reactions should flip sign.
5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [4.0, 0.0, 5.0], [4.0, 0.0, 0.0]])
    n_nodes = len(node_coords)
    props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000166}
    elements = [{'node_i': 0, 'node_j': 1, **props, 'local_z': np.array([1.0, 0.0, 0.0])}, {'node_i': 1, 'node_j': 2, **props, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 3, **props, 'local_z': np.array([1.0, 0.0, 0.0])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1]}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.all(u0 == 0.0)
    assert np.all(r0 == 0.0)
    loads1 = {1: np.array([1000.0, 2000.0, 3000.0, 400.0, 500.0, 600.0])}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert not np.all(u1 == 0.0)
    assert not np.all(r1 == 0.0)
    loads2 = {key: 2 * val for (key, val) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads3 = {key: -val for (key, val) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    total_applied_force = np.zeros(3)
    for (node_idx, load_vec) in loads1.items():
        total_applied_force += load_vec[:3]
    total_reaction_force = np.zeros(3)
    for i in range(n_nodes):
        total_reaction_force += r1[i * 6:i * 6 + 3]
    assert np.allclose(total_applied_force + total_reaction_force, np.zeros(3), atol=1e-09)
    total_applied_moment = np.zeros(3)
    for (node_idx, load_vec) in loads1.items():
        pos = node_coords[node_idx]
        force = load_vec[:3]
        moment = load_vec[3:]
        total_applied_moment += moment + np.cross(pos, force)
    total_reaction_moment = np.zeros(3)
    for i in range(n_nodes):
        pos = node_coords[i]
        reaction_force = r1[i * 6:i * 6 + 3]
        reaction_moment = r1[i * 6 + 3:i * 6 + 6]
        total_reaction_moment += reaction_moment + np.cross(pos, reaction_force)
    assert np.allclose(total_applied_moment + total_reaction_moment, np.zeros(3), atol=1e-09)