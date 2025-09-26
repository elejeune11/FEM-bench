def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    import numpy as np
    from math import sqrt
    L = 10.0
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 0.0001
    J = 0.0002
    n_elements = 10
    n_nodes = n_elements + 1
    beam_dir = np.array([1, 1, 1])
    beam_dir = beam_dir / np.linalg.norm(beam_dir)
    node_coords = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords[i] = i * (L / n_elements) * beam_dir
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 0, 1])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    force_magnitude = 1000.0
    force_dir = np.cross(beam_dir, np.array([0, 0, 1]))
    force_dir = force_dir / np.linalg.norm(force_dir)
    force_vector = force_magnitude * force_dir
    nodal_loads = {n_nodes - 1: [force_vector[0], force_vector[1], force_vector[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    beam_length = L * sqrt(3)
    analytical_deflection = force_magnitude * beam_length ** 3 / (3 * E * I_y)
    tip_disp_index = 6 * (n_nodes - 1)
    tip_displacement = u[tip_disp_index:tip_disp_index + 3]
    computed_deflection = np.linalg.norm(tip_displacement)
    assert abs(computed_deflection - analytical_deflection) / analytical_deflection < 0.01

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    import numpy as np
    node_coords = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0], [0, 0, 5], [5, 0, 5], [5, 5, 5], [0, 5, 5]])
    elements = []
    (E, nu, A, I_y, I_z, J) = (200000000000.0, 0.3, 0.02, 0.0001, 0.0001, 0.0002)
    connections = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for (i, j) in connections:
        elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 0, 1])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, {})
    assert np.allclose(u1, 0)
    assert np.allclose(r1, 0)
    nodal_loads = {4: [10000.0, 0, 0, 0, 0, 0], 5: [0, -5000.0, 0, 0, 0, 0], 6: [0, 0, 20000.0, 0, 0, 0], 7: [0, 0, 0, 50, 0, 0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u2, 0)
    assert not np.allclose(r2, 0)
    doubled_loads = {k: [2 * x for x in v] for (k, v) in nodal_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, doubled_loads)
    assert np.allclose(u3, 2 * u2, rtol=1e-10)
    assert np.allclose(r3, 2 * r2, rtol=1e-10)
    negated_loads = {k: [-x for x in v] for (k, v) in nodal_loads.items()}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, negated_loads)
    assert np.allclose(u4, -u2, rtol=1e-10)
    assert np.allclose(r4, -r2, rtol=1e-10)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for (node, load) in nodal_loads.items():
        total_force += load[:3]
        total_moment += load[3:]
        moment_arm = node_coords[node] - node_coords[0]
        total_moment += np.cross(moment_arm, load[:3])
    reaction_force = np.zeros(3)
    reaction_moment = np.zeros(3)
    for (node, bc) in boundary_conditions.items():
        node_start = 6 * node
        reaction = r2[node_start:node_start + 6]
        reaction_force += reaction[:3]
        reaction_moment += reaction[3:]
        moment_arm = node_coords[node] - node_coords[0]
        reaction_moment += np.cross(moment_arm, reaction[:3])
    assert np.allclose(total_force + reaction_force, 0, atol=1e-10)
    assert np.allclose(total_moment + reaction_moment, 0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff)."""
    import numpy as np
    import pytest
    node_coords = np.array([[0, 0, 0], [5, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0, 0, 1])}]
    boundary_conditions = {}
    nodal_loads = {1: [10000.0, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)