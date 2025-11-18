def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Verify elastic critical load factors match Euler theory for cantilever circular columns."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elements = 10
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4
        J = 2 * I
        I_rho = 2 * I
        for L in lengths:
            z = np.linspace(0, L, n_elements + 1)
            nodes = np.zeros((n_elements + 1, 3))
            nodes[:, 2] = z
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
            bcs = {0: [True] * 6}
            P = 1.0
            loads = {n_elements: [0, 0, -P, 0, 0, 0]}
            (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
            P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_cr_fem = lambda_cr * P
            rel_error = abs(P_cr_fem - P_cr_euler) / P_cr_euler
            assert rel_error < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Verify critical load and mode shape invariance under rigid rotation."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    h = 0.2
    b = 0.1
    n_elements = 10
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = min(Iy, Iz) / 3
    I_rho = Iy + Iz
    nodes_base = np.zeros((n_elements + 1, 3))
    nodes_base[:, 2] = np.linspace(0, L, n_elements + 1)
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
    bcs = {0: [True] * 6}
    loads_base = {n_elements: [0, 0, -1.0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads_base)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
    nodes_rot = nodes_base @ R.T
    elements_rot = []
    for i in range(n_elements):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': R @ [0, 1, 0]})
    loads_rot = {n_elements: np.concatenate((R @ [0, 0, -1.0], [0, 0, 0]))}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    n_nodes = n_elements + 1
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    assert abs(lambda_rot - lambda_base) / lambda_base < 1e-10
    mode_rot_transformed = T @ mode_base
    correlation = np.abs(np.dot(mode_rot, mode_rot_transformed)) / (np.linalg.norm(mode_rot) * np.linalg.norm(mode_rot_transformed))
    assert correlation > 0.999

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify mesh convergence to analytical Euler buckling load."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    r = 0.5
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    I_rho = 2 * I
    P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32, 64]
    rel_errors = []
    for n_elements in n_elements_list:
        nodes = np.zeros((n_elements + 1, 3))
        nodes[:, 2] = np.linspace(0, L, n_elements + 1)
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
        bcs = {0: [True] * 6}
        loads = {n_elements: [0, 0, -1.0, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_cr_fem = lambda_cr
        rel_error = abs(P_cr_fem - P_cr_euler) / P_cr_euler
        rel_errors.append(rel_error)
    for i in range(len(rel_errors) - 1):
        ratio = rel_errors[i] / rel_errors[i + 1]
        assert ratio > 3.5
    assert rel_errors[-1] < 1e-06