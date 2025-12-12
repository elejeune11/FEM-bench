def test_simple_beam_discretized_axis_111(fcn):
    import numpy as np
    from collections import Sequence
    from math import sqrt
    L = 10.0
    n_elements = 10
    n_nodes = n_elements + 1
    dx = L / n_elements / sqrt(3)
    node_coords = np.array([[i * dx, i * dx, i * dx] for i in range(n_nodes)])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 0.0001
    J = 0.0002
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0, 1, 0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_tip = 1000.0
    load_dir = np.array([1, -1, 0])
    load_dir /= np.linalg.norm(load_dir)
    nodal_loads = {n_nodes - 1: [F_tip * load_dir[0], F_tip * load_dir[1], F_tip * load_dir[2], 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    I_perp = I_y
    L_eff = L
    delta_analytical = F_tip * L_eff ** 3 / (3 * E * I_perp)
    tip_node_idx = n_nodes - 1
    tip_disp = u[6 * tip_node_idx:6 * tip_node_idx + 3]
    disp_magnitude = np.linalg.norm(tip_disp)
    assert abs(disp_magnitude - delta_analytical) / delta_analytical < 0.05

def test_complex_geometry_and_basic_loading(fcn):
    import numpy as np
    from collections import Sequence
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 0, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 0, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u1, 0)
    assert np.allclose(r1, 0)
    nodal_loads = {1: [100, 200, 300, 10, 20, 30], 2: [400, 500, 600, 40, 50, 60], 3: [700, 800, 900, 70, 80, 90]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u2, 0)
    assert not np.allclose(r2, 0)
    nodal_loads_double = {k: [2 * x for x in v] for (k, v) in nodal_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u3, 2 * u2)
    assert np.allclose(r3, 2 * r2)
    nodal_loads_neg = {k: [-x for x in v] for (k, v) in nodal_loads.items()}
    (u4, r4) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u4, -u2)
    assert np.allclose(r4, -r2)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for (node_idx, loads) in nodal_loads.items():
        total_applied_force += np.array(loads[:3])
        total_applied_moment += np.array(loads[3:])
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for (node_idx, bc) in boundary_conditions.items():
        if any((b == 1 for b in bc)):
            idx_start = 6 * node_idx
            for i in range(6):
                if bc[i] == 1:
                    if i < 3:
                        total_reaction_force[i] += r2[idx_start + i]
                    else:
                        total_reaction_moment[i - 3] += r2[idx_start + i]
    assert np.allclose(total_applied_force + total_reaction_force, 0, atol=1e-10)
    assert np.allclose(total_applied_moment + total_reaction_moment, 0, atol=1e-10)