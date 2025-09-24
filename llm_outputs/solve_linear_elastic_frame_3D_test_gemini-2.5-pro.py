def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    n_elem = 10
    n_nodes = n_elem + 1
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-06
    I_z = 8.333e-06
    J = 1.406e-05
    P = 1000.0
    beam_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    node_coords = np.array([i * (L / n_elem) * beam_axis for i in range(n_nodes)])
    local_z_dir = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_dir})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    local_y_dir = np.cross(local_z_dir, beam_axis)
    force_vector = P * local_y_dir
    nodal_loads = {n_nodes - 1: np.hstack([force_vector, np.zeros(3)])}
    analytical_deflection_mag = P * L ** 3 / (3 * E * I_z)
    expected_tip_displacement = analytical_deflection_mag * local_y_dir
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node_idx = n_nodes - 1
    computed_tip_displacement = u[tip_node_idx * 6:tip_node_idx * 6 + 3]
    assert np.allclose(computed_tip_displacement, expected_tip_displacement, rtol=0.05)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
Suggested Test stages:
1. Zero loads -> All displacements and reactions should be zero.
2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
3. Double the loads -> Displacements and reactions should double (linearity check).
4. Negate the original loads -> Displacements and reactions should flip sign.
5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 3.0, 0.0]])
    n_nodes = node_coords.shape[0]
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.406e-05, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.all(u_zero == 0)
    assert np.all(r_zero == 0)
    nodal_loads_1 = {2: [1000.0, 2000.0, 3000.0, 400.0, 500.0, 600.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert not np.all(u1 == 0)
    assert not np.all(r1 == 0)
    nodal_loads_2 = {k: [2 * val for val in v] for (k, v) in nodal_loads_1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    nodal_loads_3 = {k: [-val for val in v] for (k, v) in nodal_loads_1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_3)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    total_applied_forces = np.zeros(3)
    total_applied_moments_origin = np.zeros(3)
    for (node_idx, loads) in nodal_loads_1.items():
        pos = node_coords[node_idx]
        forces = np.array(loads[:3])
        moments = np.array(loads[3:])
        total_applied_forces += forces
        total_applied_moments_origin += moments + np.cross(pos, forces)
    r1_reshaped = r1.reshape((n_nodes, 6))
    total_reaction_forces = np.sum(r1_reshaped[:, :3], axis=0)
    total_reaction_moments_origin = np.zeros(3)
    for i in range(n_nodes):
        pos = node_coords[i]
        force = r1_reshaped[i, :3]
        moment = r1_reshaped[i, 3:]
        total_reaction_moments_origin += moment + np.cross(pos, force)
    assert np.allclose(total_applied_forces + total_reaction_forces, 0, atol=1e-09)
    assert np.allclose(total_applied_moments_origin + total_reaction_moments_origin, 0, atol=1e-09)