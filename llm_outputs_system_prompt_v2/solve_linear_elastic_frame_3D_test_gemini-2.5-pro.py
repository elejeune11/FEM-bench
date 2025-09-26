def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    radius = 0.1
    A = np.pi * radius ** 2
    I = np.pi * radius ** 4 / 4.0
    J = np.pi * radius ** 4 / 2.0
    P_mag = 10000.0
    n_elements = 10
    n_nodes = n_elements + 1
    beam_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    v_up = np.array([0.0, 0.0, 1.0])
    local_y_dir = np.cross(v_up, beam_axis)
    local_y_dir /= np.linalg.norm(local_y_dir)
    local_z_dir = np.cross(beam_axis, local_y_dir)
    local_z_dir /= np.linalg.norm(local_z_dir)
    node_coords = np.array([i * (L / n_elements) * beam_axis for i in range(n_nodes)])
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z_dir})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    load_force_vec = P_mag * local_y_dir
    nodal_loads = {n_nodes - 1: np.hstack([load_force_vec, [0, 0, 0]])}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    analytical_deflection_mag = P_mag * L ** 3 / (3 * E * I)
    analytical_deflection_vec = analytical_deflection_mag * local_y_dir
    tip_node_idx = n_nodes - 1
    computed_deflection_vec = u[tip_node_idx * 6:tip_node_idx * 6 + 3]
    assert np.linalg.norm(computed_deflection_vec) < analytical_deflection_mag
    assert np.allclose(computed_deflection_vec, analytical_deflection_vec, rtol=0.01)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    (L, W, H) = (2.0, 1.0, 1.5)
    node_coords = np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0], [0, 0, H], [L, 0, H], [L, W, H], [0, W, H]])
    n_nodes = len(node_coords)
    E = 200000000000.0
    nu = 0.3
    (b, t) = (0.1, 0.01)
    A = b ** 2 - (b - 2 * t) ** 2
    I = b ** 4 / 12 - (b - 2 * t) ** 4 / 12
    J = 2 * t * (b - t) ** 3
    props = {'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}
    elements = [{'node_i': 0, 'node_j': 4, 'local_z': [1, 0, 0], **props}, {'node_i': 1, 'node_j': 5, 'local_z': [1, 0, 0], **props}, {'node_i': 2, 'node_j': 6, 'local_z': [1, 0, 0], **props}, {'node_i': 3, 'node_j': 7, 'local_z': [1, 0, 0], **props}, {'node_i': 4, 'node_j': 5, 'local_z': [0, 0, 1], **props}, {'node_i': 5, 'node_j': 6, 'local_z': [0, 0, 1], **props}, {'node_i': 6, 'node_j': 7, 'local_z': [0, 0, 1], **props}, {'node_i': 7, 'node_j': 4, 'local_z': [0, 0, 1], **props}]
    boundary_conditions = {i: [1] * 6 for i in range(4)}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.all(u0 == 0)
    assert np.all(r0 == 0)
    load_node = 6
    load_vector = np.array([1000.0, 2000.0, -5000.0, 500.0, -300.0, 100.0])
    nodal_loads_1 = {load_node: load_vector}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    nodal_loads_2 = {load_node: 2.0 * load_vector}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u2, 2.0 * u1, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, atol=1e-09)
    nodal_loads_3 = {load_node: -1.0 * load_vector}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_3)
    assert np.allclose(u3, -1.0 * u1, atol=1e-09)
    assert np.allclose(r3, -1.0 * r1, atol=1e-09)
    F_applied = np.zeros(6 * n_nodes)
    for (node_idx, loads) in nodal_loads_1.items():
        F_applied[node_idx * 6:node_idx * 6 + 6] = loads
    F_total = F_applied + r1
    total_forces = np.sum(F_total.reshape(n_nodes, 6)[:, :3], axis=0)
    assert np.allclose(total_forces, 0, atol=1e-06)
    total_moments = np.zeros(3)
    for i in range(n_nodes):
        node_loads = F_total[i * 6:i * 6 + 6]
        forces = node_loads[:3]
        moments = node_loads[3:]
        position = node_coords[i]
        total_moments += moments + np.cross(position, forces)
    assert np.allclose(total_moments, 0, atol=1e-06)