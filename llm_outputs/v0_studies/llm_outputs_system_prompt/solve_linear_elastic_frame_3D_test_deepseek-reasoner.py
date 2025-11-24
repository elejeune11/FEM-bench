def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    I = 0.0001
    A = 0.01
    nu = 0.3
    J = 0.0002
    n_elements = 10
    n_nodes = n_elements + 1
    beam_direction = np.array([1.0, 1.0, 1.0])
    beam_direction /= np.linalg.norm(beam_direction)
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i] = i * L / n_elements * beam_direction
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': None})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_magnitude = 1000.0
    force_direction = np.array([1.0, -1.0, 0.0])
    force_direction /= np.linalg.norm(force_direction)
    force_vector = force_magnitude * force_direction
    nodal_loads = {n_nodes - 1: [force_vector[0], force_vector[1], force_vector[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_node_idx = n_nodes - 1
    tip_displacement = u[6 * tip_node_idx:6 * tip_node_idx + 3]
    analytical_deflection = force_magnitude * L ** 3 / (3 * E * I)
    displacement_magnitude = np.linalg.norm(tip_displacement)
    assert abs(displacement_magnitude - analytical_deflection) / analytical_deflection < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0], [2.5, 2.5, 3.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 1, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 2, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 3, 'node_j': 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [1, 1, 1, 0, 0, 0], 3: [0, 1, 1, 0, 0, 0]}
    n_nodes = len(node_coords)
    nodal_loads_zero = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert np.allclose(u_zero, 0.0)
    assert np.allclose(r_zero, 0.0)
    nodal_loads_original = {2: [1000.0, -500.0, 2000.0, 500.0, -300.0, 400.0], 4: [-800.0, 600.0, 1500.0, -200.0, 450.0, -350.0]}
    (u_original, r_original) = fcn(node_coords, elements, boundary_conditions, nodal_loads_original)
    assert not np.allclose(u_original, 0.0)
    assert not np.allclose(r_original, 0.0)
    nodal_loads_double = {}
    for (node, loads) in nodal_loads_original.items():
        nodal_loads_double[node] = [2 * load for load in loads]
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2 * u_original, rtol=1e-10)
    assert np.allclose(r_double, 2 * r_original, rtol=1e-10)
    nodal_loads_negated = {}
    for (node, loads) in nodal_loads_original.items():
        nodal_loads_negated[node] = [-load for load in loads]
    (u_negated, r_negated) = fcn(node_coords, elements, boundary_conditions, nodal_loads_negated)
    assert np.allclose(u_negated, -u_original, rtol=1e-10)
    assert np.allclose(r_negated, -r_original, rtol=1e-10)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for node in range(n_nodes):
        node_loads = nodal_loads_original.get(node, [0.0] * 6)
        forces = node_loads[:3]
        moments = node_loads[3:]
        total_force += np.array(forces)
        node_pos = node_coords[node]
        total_moment += np.array(moments) + np.cross(node_pos, forces)
    for node in range(n_nodes):
        if node in boundary_conditions:
            bc = boundary_conditions[node]
            for dof in range(6):
                if bc[dof] == 1:
                    reaction_index = 6 * node + dof
                    reaction_value = r_original[reaction_index]
                    if dof < 3:
                        total_force[dof] += reaction_value
                        node_pos = node_coords[node]
                        if dof == 0:
                            total_moment[1] += -reaction_value * node_pos[2]
                            total_moment[2] += reaction_value * node_pos[1]
                        elif dof == 1:
                            total_moment[0] += reaction_value * node_pos[2]
                            total_moment[2] += -reaction_value * node_pos[0]
                        else:
                            total_moment[0] += -reaction_value * node_pos[1]
                            total_moment[1] += reaction_value * node_pos[0]
                    else:
                        total_moment[dof - 3] += reaction_value
    assert np.allclose(total_force, 0.0, atol=1e-10)
    assert np.allclose(total_moment, 0.0, atol=1e-10)