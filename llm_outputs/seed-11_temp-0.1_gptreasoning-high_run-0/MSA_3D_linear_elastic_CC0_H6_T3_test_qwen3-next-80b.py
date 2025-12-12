def test_simple_beam_discretized_axis_111(fcn):
    import numpy as np
    from collections import Sequence
    from math import sqrt
    L = 10.0
    n_elements = 10
    dx = L / n_elements
    node_coords = np.array([[i * dx / sqrt(3), i * dx / sqrt(3), i * dx / sqrt(3)] for i in range(n_elements + 1)])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 0.0001
    J = 0.0002
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [1, -1, 0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_tip = 1000.0
    load_dir = np.array([1, 1, -2])
    load_dir /= np.linalg.norm(load_dir)
    nodal_loads = {n_elements: [F_tip * load_dir[0], F_tip * load_dir[1], F_tip * load_dir[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I_eff = I_y
    delta_analytical = F_tip * L ** 3 / (3 * E * I_eff)
    tip_u = u[6 * n_elements:6 * n_elements + 3]
    delta_computed = np.linalg.norm(tip_u)
    assert abs(delta_computed - delta_analytical) / delta_analytical < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    import numpy as np
    from collections import Sequence
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 0, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 0, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u_zero, r_zero) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u_zero, 0)
    assert np.allclose(r_zero, 0)
    nodal_loads = {1: [1000, 0, 0, 0, 50, 0], 2: [0, 2000, 0, 0, 0, -100], 3: [0, 0, 3000, 25, 0, 0]}
    (u_nonzero, r_nonzero) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u_nonzero, 0)
    assert not np.allclose(r_nonzero, 0)
    nodal_loads_double = {k: [2 * x for x in v] for (k, v) in nodal_loads.items()}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u_double, 2 * u_nonzero)
    assert np.allclose(r_double, 2 * r_nonzero)
    nodal_loads_neg = {k: [-x for x in v] for (k, v) in nodal_loads.items()}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u_nonzero)
    assert np.allclose(r_neg, -r_nonzero)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node_idx, loads) in nodal_loads.items():
        total_applied_force += np.array(loads[:3])
        total_applied_moment += np.array(loads[3:])
    fixed_reactions = r_nonzero[:6]
    total_reaction_force = fixed_reactions[:3]
    total_reaction_moment = fixed_reactions[3:]
    assert np.allclose(total_reaction_force, -total_applied_force)
    applied_moment_due_to_forces = np.zeros(3)
    for (node_idx, loads) in nodal_loads.items():
        force = np.array(loads[:3])
        pos = node_coords[node_idx]
        moment_arm = np.cross(pos, force)
        applied_moment_due_to_forces += moment_arm
    total_applied_moment_about_fixed = applied_moment_due_to_forces + total_applied_moment
    assert np.allclose(total_reaction_moment, -total_applied_moment_about_fixed, atol=1e-06)